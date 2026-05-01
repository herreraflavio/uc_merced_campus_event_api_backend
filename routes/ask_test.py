# ask_baseline.py

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

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------

CONTENT_API_URL = os.getenv("CONTENT_API_URL", "http://10.56.184.54:8080/contentAPIURL")
MODEL_NAME = os.getenv("BASELINE_OPENAI_MODEL_NAME", os.getenv("OPENAI_MODEL_NAME", "gpt-4o"))

CONTENT_API_TIMEOUT_SECONDS = int(os.getenv("CONTENT_API_TIMEOUT_SECONDS", "15"))

# GPT-4o default text pricing per 1M tokens.
# Override these in .env if you change model/pricing.
INPUT_COST_PER_1M = float(
    os.getenv("BASELINE_INPUT_COST_PER_1M", os.getenv("OPENAI_INPUT_COST_PER_1M", "2.50"))
)
OUTPUT_COST_PER_1M = float(
    os.getenv("BASELINE_OUTPUT_COST_PER_1M", os.getenv("OPENAI_OUTPUT_COST_PER_1M", "10.00"))
)


# ------------------------------------------------------------------------------
# USAGE / COST HELPERS
# ------------------------------------------------------------------------------

def empty_usage():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "context_tokens": 0,
        "total_tokens": 0,
    }


def get_openai_usage(resp):
    usage = getattr(resp, "usage", None)

    if not usage:
        return empty_usage()

    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", None)

    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        # Actual context sent to the model.
        "context_tokens": input_tokens,
        "total_tokens": total_tokens,
    }


def calculate_cost(usage):
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0

    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
    total_cost = input_cost + output_cost

    return {
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(total_cost, 8),
        "input_cost_per_1m_tokens": INPUT_COST_PER_1M,
        "output_cost_per_1m_tokens": OUTPUT_COST_PER_1M,
        "currency": "USD",
    }


def build_empty_response(message, status_code=200):
    usage = empty_usage()
    cost = calculate_cost(usage)

    return jsonify({
        "ai_overview": message,
        "citations": [],
        "ranked_item_ids": [],
        "usage": usage,
        "cost": cost,
        "estimated_cost_usd": cost["total_cost_usd"],
        "model": MODEL_NAME,
    }), status_code


# ------------------------------------------------------------------------------
# RESPONSE HELPERS
# ------------------------------------------------------------------------------

def get_citation_snippet(item):
    snippet = "Location not specified"

    if isinstance(item.get("label"), dict) and item["label"].get("name"):
        snippet = str(item["label"].get("name")).strip()
    elif item.get("location"):
        snippet = str(item.get("location")).strip()
    elif item.get("host"):
        snippet = str(item.get("host")).strip()

    return snippet


def parse_ranked_ids(raw_output, valid_item_ids):
    tag_match = re.search(
        r"\[IDS:\s*(.*?)\]",
        raw_output,
        re.IGNORECASE | re.DOTALL,
    )

    if not tag_match:
        return []

    found_ids = [
        x.strip()
        for x in tag_match.group(1).split(",")
        if x.strip()
    ]

    return [
        x for x in found_ids
        if x in valid_item_ids
    ]


# ------------------------------------------------------------------------------
# BASELINE LLM ROUTE
# ------------------------------------------------------------------------------

@ask_test_bp.route("/ai_baseline", methods=["POST"])
def ask_ai():
    data = request.get_json(silent=True) or {}

    query = str(data.get("query", "")).strip()
    item_ids = data.get("item_ids", [])

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if not isinstance(item_ids, list):
        return jsonify({"error": "item_ids must be an array"}), 400

    if not item_ids:
        return build_empty_response("No item_ids were provided in the request.")

    try:
        content_resp = requests.get(
            CONTENT_API_URL,
            timeout=CONTENT_API_TIMEOUT_SECONDS
        )
        content_resp.raise_for_status()

        content_json = content_resp.json()
        pages = content_json.get("pages", [])

        if not isinstance(pages, list):
            return jsonify({"error": "Invalid content API response"}), 500

        item_id_set = set(item_ids)

        # Baseline intentionally does NOT condense or truncate.
        # This feeds the full matching knowledge base items into GPT-4o.
        valid_items = [
            p for p in pages
            if p.get("id") in item_id_set
        ]

        if not valid_items:
            return build_empty_response(
                "I could not find any matching items for the item_ids you sent."
            )

        valid_item_ids = {
            item.get("id")
            for item in valid_items
            if item.get("id")
        }

        prompt_content = f"""
You are an expert campus informant reasoning model.
Your task is to analyze the user's query against the provided full JSON array of campus items.

USER QUERY:
{query}

AVAILABLE ITEMS:
{json.dumps(valid_items, indent=2, ensure_ascii=False, default=str)}

INSTRUCTIONS:
1. Analyze the campus items carefully.
2. Provide a comprehensive overview in 1-3 sentences summarizing the most relevant items found.
3. Rank up to 10 relevant item IDs from best to worst match.
4. Only rank item IDs that appear in AVAILABLE ITEMS.
5. Output your ranked IDs EXACTLY in this format at the very end of your response:
[IDS: id1, id2, id3]
""".strip()

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.3,
        )

        raw_output = (resp.choices[0].message.content or "").strip()

        usage = get_openai_usage(resp)
        cost = calculate_cost(usage)

        ranked_item_ids = parse_ranked_ids(raw_output, valid_item_ids)

        if not ranked_item_ids:
            ranked_item_ids = [
                item.get("id")
                for item in valid_items[:10]
                if item.get("id")
            ]

        ai_overview = re.sub(
            r"\[IDS:.*?\]",
            "",
            raw_output,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        if not ai_overview:
            ai_overview = "Here are the top matches based on your search."

        citations = []

        for pid in ranked_item_ids:
            matched_item = next(
                (item for item in valid_items if item.get("id") == pid),
                None,
            )

            if not matched_item:
                continue

            citations.append({
                "page_id": pid,
                "title": matched_item.get("title", ""),
                "snippet": get_citation_snippet(matched_item),
            })

        return jsonify({
            "ai_overview": ai_overview,
            "citations": citations,
            "ranked_item_ids": ranked_item_ids,
            "usage": usage,
            "cost": cost,
            "estimated_cost_usd": cost["total_cost_usd"],
            "model": MODEL_NAME,
        }), 200

    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch content API data",
            "details": str(e),
        }), 502

    except Exception as e:
        return jsonify({
            "error": "Failed to generate AI response",
            "details": str(e),
        }), 500