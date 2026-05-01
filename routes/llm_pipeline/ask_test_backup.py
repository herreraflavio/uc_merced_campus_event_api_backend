import os
import json
import re
import requests
from flask import request, jsonify, Blueprint
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ask_test_bp = Blueprint("ask_test", __name__)

# CONFIG
CONTENT_API_URL = os.getenv("CONTENT_API_URL", "http://10.34.6.109:8080/contentAPIURL")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "o3-mini")

DEFAULT_MAX_TOKENS = 800


def empty_usage(context_tokens):
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "context_tokens": context_tokens
    }


def get_openai_usage(resp, context_tokens):
    usage = getattr(resp, "usage", None)

    if not usage:
        return empty_usage(context_tokens)

    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "context_tokens": context_tokens
    }


def condense_item(item: dict) -> dict:
    condensed = {
        "id": item.get("id"),
        "title": item.get("title", ""),
        "type": item.get("type", ""),
        "host": item.get("host", ""),
        "tags": item.get("tags", []),
    }

    desc = item.get("description", "") or ""
    condensed["description"] = desc[:300].strip()

    nested = item.get("nested_content")
    if nested:
        nested_str = json.dumps(nested, separators=(",", ":"))
        nested_str = re.sub(r"https?://\S+", "", nested_str)
        condensed["nested_content"] = nested_str[:600]

    return condensed


@ask_test_bp.route("/ai_baseline", methods=["POST"])
def ask_ai():
    data = request.get_json(silent=True) or {}

    query = str(data.get("query", "")).strip()
    item_ids = data.get("item_ids", [])

    max_tokens = data.get("max_tokens", DEFAULT_MAX_TOKENS)

    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        return jsonify({"error": "max_tokens must be an integer"}), 400

    if max_tokens <= 0:
        return jsonify({"error": "max_tokens must be greater than 0"}), 400

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if not isinstance(item_ids, list):
        return jsonify({"error": "item_ids must be an array"}), 400

    if not item_ids:
        return jsonify({
            "ai_overview": "No item_ids were provided in the request.",
            "citations": [],
            "ranked_item_ids": [],
            "usage": empty_usage(max_tokens)
        }), 200

    try:
        content_resp = requests.get(CONTENT_API_URL, timeout=15)
        content_resp.raise_for_status()
        pages = content_resp.json().get("pages", [])

        valid_items = [
            p for p in pages
            if p.get("id") in item_ids
        ]

        if not valid_items:
            return jsonify({
                "ai_overview": "I could not find any matching items for the item_ids you sent.",
                "citations": [],
                "ranked_item_ids": [],
                "usage": empty_usage(max_tokens)
            }), 200

        condensed_items = [
            condense_item(item)
            for item in valid_items
        ]

        prompt_content = f"""
You are an expert campus informant reasoning model.
Your task is to analyze the user's query against the provided JSON array of campus items.

USER QUERY: "{query}"

AVAILABLE ITEMS:
{json.dumps(condensed_items, indent=2)}

INSTRUCTIONS:
1. Think step-by-step to find the items that best match the user's query.
2. Provide a comprehensive overview (1-3 sentences) summarizing the most relevant items found.
3. Rank up to 10 relevant item IDs from best to worst match.
4. Output your ranked IDs EXACTLY in this format at the very end of your response:
[IDS: id1, id2, id3]
"""

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt_content}
            ],
            max_completion_tokens=max_tokens
        )

        raw_output = (resp.choices[0].message.content or "").strip()

        usage = get_openai_usage(resp, max_tokens)

        ranked_item_ids = []

        tag_match = re.search(
            r"\[IDS:\s*(.*?)\]",
            raw_output,
            re.IGNORECASE | re.DOTALL
        )

        if tag_match:
            found_ids = [
                x.strip()
                for x in tag_match.group(1).split(",")
                if x.strip()
            ]

            ranked_item_ids = [
                x for x in found_ids
                if x in item_ids
            ]

        if not ranked_item_ids:
            ranked_item_ids = [
                item["id"]
                for item in condensed_items[:10]
            ]

        ai_overview = re.sub(
            r"\[IDS:.*?\]",
            "",
            raw_output,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()

        if not ai_overview:
            ai_overview = "Here are the top matches based on your search."

        citations = []

        for pid in ranked_item_ids:
            matched_item = next(
                (item for item in valid_items if item.get("id") == pid),
                None
            )

            if not matched_item:
                continue

            snippet = "Location not specified"

            if isinstance(matched_item.get("label"), dict) and matched_item["label"].get("name"):
                snippet = str(matched_item["label"].get("name")).strip()
            elif matched_item.get("location"):
                snippet = str(matched_item.get("location")).strip()
            elif matched_item.get("host"):
                snippet = str(matched_item.get("host")).strip()

            citations.append({
                "page_id": pid,
                "title": matched_item.get("title", ""),
                "snippet": snippet,
            })

        return jsonify({
            "ai_overview": ai_overview,
            "citations": citations,
            "ranked_item_ids": ranked_item_ids,
            "usage": usage
        }), 200

    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch content API data",
            "details": str(e)
        }), 502

    except Exception as e:
        return jsonify({
            "error": "Failed to generate AI response",
            "details": str(e)
        }), 500