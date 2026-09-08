from flask import Blueprint, jsonify, request
import json
import logging
from typing import Any

MODEL_NAME = "gpt-4o"
MAX_IMAGES = 4

logger = logging.getLogger(__name__)


def create_autofill_ai_blueprint(client) -> Blueprint:
    """
    Create a Flask blueprint for wildlife AI autofill.

    Pass the existing OpenAI client instance into this function so this module
    reuses the same configured client as the rest of the application.

    Example:
        from autofill_ai import create_autofill_ai_blueprint

        app.register_blueprint(create_autofill_ai_blueprint(client))
    """
    bp = Blueprint("autofill_ai", __name__)

    @bp.route("/autofill_ai", methods=["POST"])
    def autofill_ai():
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        if data.get("ai_data_sharing_consent") is not True:
            return jsonify({
                "error": {
                    "code": "ai_data_sharing_consent_required",
                    "message": "AI Autofill data sharing consent is required.",
                }
            }), 400

        image_urls = data.get("image_urls", [])
        if not isinstance(image_urls, list):
            return jsonify({"error": "image_urls must be an array"}), 400

        image_urls = [
            url.strip()
            for url in image_urls
            if isinstance(url, str) and url.strip()
        ]

        if not image_urls:
            return jsonify({"error": "At least one image URL is required"}), 400

        if len(image_urls) > MAX_IMAGES:
            image_urls = image_urls[:MAX_IMAGES]

        existing_title = _clean_string(data.get("title"))
        existing_description = _clean_string(data.get("description"))
        existing_tags = _clean_tags(data.get("tags"))

        context = {
            "type": _clean_string(data.get("type")) or "wildlife",
            "title": existing_title,
            "description": existing_description,
            "tags": existing_tags,
            "subtitle": _clean_string(data.get("subtitle")),
            "host": _clean_string(data.get("host")),
            "location_id": data.get("location_id"),
            "start": data.get("start"),
            "end": data.get("end"),
            "source_url": _clean_string(data.get("source_url")),
        }

        system_prompt = """
You generate concise autofill metadata for wildlife reports submitted from a
university campus map.

Analyze the supplied wildlife image or images and use the report context only
as supporting information.

Return ONLY valid JSON with exactly these keys:

{
  "title": string | null,
  "description": string | null,
  "tags": string[] | null,
  "image_alt_text": string | null
}

Rules:
- Describe only what can reasonably be inferred from the images and context.
- Do not claim an exact species when visual evidence is insufficient.
- If uncertain, use a broader identification such as "bird", "snake",
  "deer", "insect", or "unknown wildlife".
- Keep title short and useful.
- Keep description factual and concise, normally 1-2 sentences.
- tags must be lowercase, short, and useful for search/filtering.
- Use hyphens for multi-word tags when appropriate.
- image_alt_text should describe the visible wildlife and relevant scene
  for accessibility. Do not start it with "Image of" or "Photo of".
- If title is already non-empty in the supplied context, return null for title.
- If description is already non-empty, return null for description.
- If tags already contains one or more entries, return null for tags.
- Never invent a location, date, behavior, danger level, or species certainty.
- Do not return markdown, commentary, or additional keys.
""".strip()

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Autofill the missing wildlife report fields.\n\n"
                    "Current report context:\n"
                    + json.dumps(context, ensure_ascii=False, indent=2)
                ),
            }
        ]

        for url in image_urls:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
            )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=500,
            )

            raw = (response.choices[0].message.content or "").strip()
            if not raw:
                logger.error("Wildlife autofill model returned an empty response")
                return jsonify({"error": "AI returned an empty response"}), 502

            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                logger.exception("Wildlife autofill model returned invalid JSON")
                return jsonify({"error": "AI returned invalid JSON"}), 502

            normalized = _normalize_ai_response(result)

            if not any(
                normalized.get(key) is not None
                for key in ("title", "description", "tags", "image_alt_text")
            ):
                return jsonify({
                    "error": "AI did not return any usable autofill fields"
                }), 502

            return jsonify(normalized), 200

        except Exception as exc:
            logger.exception("Wildlife autofill request failed")
            return jsonify({
                "error": "Failed to generate wildlife autofill",
                "details": str(exc),
            }), 500

    return bp


def _normalize_ai_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "title": None,
            "description": None,
            "tags": None,
            "image_alt_text": None,
        }

    title = _nullable_string(value.get("title"))
    description = _nullable_string(value.get("description"))
    image_alt_text = _nullable_string(
        value.get("image_alt_text", value.get("alt_text"))
    )

    tags_value = value.get("tags")
    tags = None

    if isinstance(tags_value, list):
        cleaned_tags = []
        seen = set()

        for tag in tags_value:
            if not isinstance(tag, str):
                continue

            cleaned = tag.strip().lower()
            if not cleaned or cleaned in seen:
                continue

            seen.add(cleaned)
            cleaned_tags.append(cleaned)

        tags = cleaned_tags or None

    return {
        "title": title,
        "description": description,
        "tags": tags,
        "image_alt_text": image_alt_text,
    }


def _nullable_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    value = value.strip()
    return value or None


def _clean_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    return value.strip()


def _clean_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    return [
        tag.strip()
        for tag in value
        if isinstance(tag, str) and tag.strip()
    ]
