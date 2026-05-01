import os
import json
import logging
from flask import Blueprint, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ios_bp = Blueprint("ios", __name__)

# ------------------------------------------------------------------------------
# LOGGING CONFIGURATION
# ------------------------------------------------------------------------------

logging.basicConfig(
    filename="ai_debug_log.txt",
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)

# Path to the un-cached pages file
PAGES_JSON_PATH = os.path.join(os.getcwd(), "pages.json")


# ------------------------------------------------------------------------------
# AI POST TYPE PROMPT CONFIGURATION
# ------------------------------------------------------------------------------

POST_TYPE_GUIDANCE = {
    "wildlife": {
        "label": "Wildlife Sighting",
        "required_tags": ["wildlife"],
        "title_style": "Name the visible animal, insect, bird, plant, or natural subject if it can be identified confidently. If uncertain, use a general title like 'Wildlife Sighting' or 'Campus Wildlife'.",
        "description_focus": "Describe what is visible, where it appears to be, and any useful observation details. Do not invent species names if the image is unclear.",
        "alt_text_focus": "Describe the animal or natural subject, its visible surroundings, and important visual details for accessibility.",
        "avoid": "Do not claim a rare species, exact species, behavior, danger level, or scientific details unless clearly visible or provided.",
    },
    "event": {
        "label": "Event",
        "required_tags": ["events"],
        "title_style": "Create a concise event-style title. If the image has a flyer or visible text, use the clearest event name from the image. If not, use a general title based on the draft.",
        "description_focus": "Summarize the event purpose, visible details, and what someone should know before attending. Do not invent times, rooms, hosts, or RSVP details.",
        "alt_text_focus": "Describe the flyer, event setup, crowd, sign, or visual content shown in the image.",
        "avoid": "Do not create fake event names, fake organizations, fake dates, fake room numbers, or fake attendance details.",
    },
    "parking_lot": {
        "label": "Parking Lot",
        "required_tags": ["parking"],
        "title_style": "Use a practical parking title, such as the lot name if provided or visible. If unknown, use 'Campus Parking Area'.",
        "description_focus": "Describe parking availability context, visible lot features, nearby landmarks, and how the place may help map users.",
        "alt_text_focus": "Describe the parking lot, cars, signs, pavement markings, and nearby landmarks.",
        "avoid": "Do not invent permit rules, prices, enforcement details, or exact lot names unless provided or clearly visible.",
    },
    "bus_stop": {
        "label": "Bus Stop",
        "required_tags": ["bus", "transportation"],
        "title_style": "Use a bus-stop title based on visible signage or nearby location. If unknown, use 'Campus Bus Stop'.",
        "description_focus": "Describe the stop, signage, shelter, curb, nearby landmark, or transit usefulness. Keep it map-oriented.",
        "alt_text_focus": "Describe the bus stop sign, shelter, bench, road, curb, and surroundings.",
        "avoid": "Do not invent route numbers, schedules, operating hours, or service names unless visible or provided.",
    },
    "dining": {
        "label": "Dining",
        "required_tags": ["dining", "food"],
        "title_style": "Use the dining location name if provided or visible. If unknown, use 'Campus Dining Spot'.",
        "description_focus": "Describe the dining place, visible food/service area, seating, entrance, or why it is useful to students.",
        "alt_text_focus": "Describe the dining area, food, counter, seating, signs, or entrance shown.",
        "avoid": "Do not invent menus, hours, prices, dietary options, or restaurant names unless visible or provided.",
    },
    "retail_store": {
        "label": "Retail Store",
        "required_tags": ["retail", "store"],
        "title_style": "Use the store name if visible or provided. If unknown, use 'Campus Retail Store'.",
        "description_focus": "Describe what kind of store or service area it appears to be and what map users can recognize visually.",
        "alt_text_focus": "Describe the storefront, shelves, signs, merchandise, entrance, or surrounding area.",
        "avoid": "Do not invent products, prices, hours, services, or store names unless visible or provided.",
    },
    "restrooms": {
        "label": "Restrooms",
        "required_tags": ["restrooms"],
        "title_style": "Use a location-oriented restroom title, such as 'Restrooms near [location]' if a location is provided.",
        "description_focus": "Describe the restroom location cue, signage, floor/area context if provided, and nearby landmark. Keep it concise and practical.",
        "alt_text_focus": "Describe restroom signage, hallway, entrance, accessibility symbols, or nearby visible landmarks.",
        "avoid": "Do not invent gender designation, accessibility features, floor number, or building name unless visible or provided.",
    },
    "sit_down_area": {
        "label": "Sit Down Area",
        "required_tags": ["seating"],
        "title_style": "Use a seating-focused title, such as 'Outdoor Seating Area' or 'Study Seating Spot' depending on visible context.",
        "description_focus": "Describe seating type, shade, indoor/outdoor context, nearby landmarks, and why it is useful.",
        "alt_text_focus": "Describe benches, tables, chairs, shade structures, people if visible, and surrounding environment.",
        "avoid": "Do not invent reservation rules, capacity, quietness, or availability unless clearly provided.",
    },
    "bike_rack": {
        "label": "Bike Rack",
        "required_tags": ["bike", "bike-rack"],
        "title_style": "Use a practical bike rack title based on nearby location if provided. If unknown, use 'Campus Bike Rack'.",
        "description_focus": "Describe the bike rack location, nearby entrance/pathway, and visible rack features.",
        "alt_text_focus": "Describe bike racks, bicycles, nearby walkway, building entrance, or surrounding area.",
        "avoid": "Do not invent security rules, capacity, or parking policy unless visible or provided.",
    },
    "viewing_spot": {
        "label": "Viewing Spot",
        "required_tags": ["viewing-spot"],
        "title_style": "Use a scenic or viewpoint-oriented title based on the visible view or nearby landmark.",
        "description_focus": "Describe what can be seen from the spot, the viewing direction if clear, and why it is useful or scenic.",
        "alt_text_focus": "Describe the view, landscape, buildings, lake, sky, pathway, or other visible scenery.",
        "avoid": "Do not invent official viewpoint names or claim visibility of landmarks that are not shown.",
    },
    "green_area": {
        "label": "Green Area / Spot",
        "required_tags": ["green-area"],
        "title_style": "Use a green-space title, such as 'Campus Green Space' or 'Shaded Lawn Area' based on the image.",
        "description_focus": "Describe grass, trees, shade, landscaping, open space, nearby walkway, or how students might recognize it.",
        "alt_text_focus": "Describe the grass, trees, plants, walkway, open area, and surrounding campus features.",
        "avoid": "Do not invent park names, official designations, rules, or amenities unless visible or provided.",
    },
}

DEFAULT_POST_TYPE_GUIDANCE = {
    "label": "Campus Map Post",
    "required_tags": ["campus"],
    "title_style": "Create a concise campus-map title based on the selected type, image, and draft.",
    "description_focus": "Describe the visible campus feature and why it is useful to map users.",
    "alt_text_focus": "Describe the image clearly for accessibility.",
    "avoid": "Do not invent precise names, rules, dates, locations, or official information unless visible or provided.",
}


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def extract_json(raw: str) -> dict:
    """
    Clean up model output and return the first {...} JSON object inside.

    Raises ValueError if no valid JSON is found.
    """
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


def safe_string(value) -> str:
    if value is None:
        return ""

    if not isinstance(value, str):
        return str(value)

    return value.strip()


def normalize_tags(tags) -> list[str]:
    if not isinstance(tags, list):
        return []

    cleaned = []

    for tag in tags:
        if not isinstance(tag, str):
            continue

        value = tag.strip().lower()
        if not value:
            continue

        value = value.replace(" ", "-")

        if value not in cleaned:
            cleaned.append(value)

    return cleaned


def get_post_type_guidance(post_type: str) -> dict:
    post_type = safe_string(post_type).lower()
    return POST_TYPE_GUIDANCE.get(post_type, DEFAULT_POST_TYPE_GUIDANCE)


def get_missing_ai_fields(data: dict) -> list[str]:
    """
    Only these fields should be auto-filled by AI:
    - title
    - description
    - tags
    - image_alt_text

    Location, start date/time, and end date/time should not be generated here.
    """
    missing = []

    if not safe_string(data.get("title")):
        missing.append("title")

    if not safe_string(data.get("description")):
        missing.append("description")

    if len(normalize_tags(data.get("tags"))) == 0:
        missing.append("tags")

    # Your current Swift request does not send image_alt_text in the request body,
    # but the response model expects it. If you later add it to the request,
    # this check will preserve existing alt text.
    existing_alt_text = safe_string(data.get("image_alt_text") or data.get("alt_text"))
    if not existing_alt_text:
        missing.append("image_alt_text")

    return missing


def build_post_type_guidance_text(post_type: str, guidance: dict) -> str:
    required_tags = guidance.get("required_tags", [])
    required_tags_text = ", ".join(required_tags)

    return f"""
Selected post type: {post_type}
Post type label: {guidance.get("label", "Campus Map Post")}

Type-specific behavior:
- Title style: {guidance.get("title_style")}
- Description focus: {guidance.get("description_focus")}
- Image alt text focus: {guidance.get("alt_text_focus")}
- Required/default tags to include when generating tags: {required_tags_text}
- Avoid: {guidance.get("avoid")}
""".strip()


def load_pages():
    """Reads pages.json directly with no caching for testing purposes."""
    if not os.path.exists(PAGES_JSON_PATH):
        return []

    try:
        with open(PAGES_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read pages.json: {e}")
        return []


def save_pages(data):
    """Writes directly to pages.json."""
    try:
        with open(PAGES_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write to pages.json: {e}")


# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------

@ios_bp.route("/ask/ai", methods=["POST"])
def ask_ai():
    """
    Expects a JSON body matching CampusAIAutofillRequest from the iOS app.

    Sends the data and image URLs to GPT-4o for auto-filling missing fields.
    The prompt adapts based on the selected post type.
    """
    print("ask ai endpoint hit")

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    image_urls = data.get("image_urls", [])
    if not isinstance(image_urls, list):
        image_urls = []

    post_type = safe_string(data.get("type")).lower() or "campus_post"
    post_type_guidance = get_post_type_guidance(post_type)
    missing_fields = get_missing_ai_fields(data)

    if not missing_fields:
        return jsonify({
            "message": "No AI-fillable fields are missing.",
            "title": data.get("title"),
            "description": data.get("description"),
            "tags": normalize_tags(data.get("tags")),
            "image_alt_text": data.get("image_alt_text") or data.get("alt_text"),
        }), 200

    post_type_guidance_text = build_post_type_guidance_text(
        post_type=post_type,
        guidance=post_type_guidance,
    )

    system_message = {
        "role": "system",
        "content": (
            "You are an AI assistant that auto-fills missing required fields for UC Merced campus map posts.\n\n"
            "You will receive a partial JSON draft and possibly one or more image URLs.\n\n"
            "Critical rules:\n"
            "1. The selected post type is authoritative. Adapt your writing to that type.\n"
            "2. Only generate values for fields listed in MISSING_FIELDS.\n"
            "3. Do not replace, rewrite, or reinterpret fields that already have user-provided values.\n"
            "4. Do not generate or modify location, location_id, geometry, start, end, host, source_url, pin_url, id, or type.\n"
            "5. If a field is not listed in MISSING_FIELDS, omit it from your JSON response.\n"
            "6. Return ONLY a valid JSON object. No markdown. No code fences. No explanations.\n"
            "7. The only allowed response keys are: title, description, tags, image_alt_text.\n"
            "8. Keep titles concise, usually 2-8 words.\n"
            "9. Keep descriptions useful for a campus map, usually 1-3 sentences.\n"
            "10. Tags must be a JSON array of lowercase strings. Prefer short kebab-case tags.\n"
            "11. When generating tags, include the required/default tags for the selected post type.\n"
            "12. Do not invent precise room numbers, event names, schedules, species, policies, prices, routes, or official names unless they are visible in the image or provided in the draft.\n"
            "13. If image evidence is unclear, be conservative and use general wording.\n"
            "14. If no image is provided, use only the text draft and selected post type. Do not pretend you saw an image.\n"
        ),
    }

    user_text = f"""
MISSING_FIELDS:
{json.dumps(missing_fields, indent=2)}

POST_TYPE_GUIDANCE:
{post_type_guidance_text}

PARTIAL_DRAFT_JSON:
{json.dumps(data, indent=2, ensure_ascii=False)}

Return a JSON object containing only the missing fields from MISSING_FIELDS.

Expected field behavior:
- title: concise display title for the selected post type.
- description: helpful map-oriented description.
- tags: lowercase tag array, including the selected type's required/default tags.
- image_alt_text: accessibility-focused description of the image. If no image is provided and there is not enough information, return a concise generic alt text based on the draft and post type.
""".strip()

    user_content = [
        {
            "type": "text",
            "text": user_text,
        }
    ]

    # Append any provided image URLs so the vision model can inspect them.
    for url in image_urls:
        if not isinstance(url, str):
            continue

        clean_url = url.strip()
        if not clean_url:
            continue

        user_content.append({
            "type": "image_url",
            "image_url": {"url": clean_url},
        })

    user_message = {
        "role": "user",
        "content": user_content,
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[system_message, user_message],
            temperature=0.25,
            max_tokens=900,
            response_format={"type": "json_object"},
        )

        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            return jsonify({"error": "Empty model response"}), 502

        try:
            result = extract_json(raw)

            # Server-side cleanup so the iOS app gets predictable values.
            cleaned_result = {}

            if "title" in missing_fields:
                title = safe_string(result.get("title"))
                if title:
                    cleaned_result["title"] = title

            if "description" in missing_fields:
                description = safe_string(result.get("description"))
                if description:
                    cleaned_result["description"] = description

            if "tags" in missing_fields:
                generated_tags = normalize_tags(result.get("tags"))

                required_tags = post_type_guidance.get("required_tags", [])
                for required_tag in required_tags:
                    cleaned_required = safe_string(required_tag).lower().replace(" ", "-")
                    if cleaned_required and cleaned_required not in generated_tags:
                        generated_tags.insert(0, cleaned_required)

                # Keep tag list compact.
                cleaned_result["tags"] = generated_tags[:6]

            if "image_alt_text" in missing_fields:
                alt_text = safe_string(
                    result.get("image_alt_text") or result.get("alt_text")
                )
                if alt_text:
                    cleaned_result["image_alt_text"] = alt_text

            logger.debug(
                "AI autofill success | type=%s | missing_fields=%s | result=%s",
                post_type,
                missing_fields,
                cleaned_result,
            )

            return jsonify(cleaned_result)

        except Exception as parse_error:
            logger.error(f"Failed to extract/clean JSON: {parse_error} | raw={raw}")
            return jsonify({
                "error": "Failed to extract JSON",
                "raw_response": raw,
            }), 500

    except Exception as e:
        logger.error(f"OpenAI API Error: {e}")
        return jsonify({"error": str(e)}), 500


@ios_bp.route("/api/pages", methods=["GET", "POST"])
def handle_pages():
    """
    GET: Returns the raw pages.json list.
    POST: Appends the new iOS post to the pages.json file.
    """
    if request.method == "GET":
        pages = load_pages()
        return jsonify(pages)

    if request.method == "POST":
        new_page = request.get_json(silent=True)
        if not new_page:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400
        print(new_page)
        pages = load_pages()
        pages.append(new_page)
        save_pages(pages)

        # The iOS app expects a 200-299 status code for success.
        return jsonify({
            "message": "Post successfully saved to pages.json",
            "id": new_page.get("id"),
        }), 201