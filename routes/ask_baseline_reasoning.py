# ask_baseline_reasoning.py

import os
import json
import re
import requests
from flask import request, jsonify, Blueprint
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ask_baseline_reasoning_bp = Blueprint(
    "ask_baseline_reasoning",
    __name__
)

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------

CONTENT_API_URL = os.getenv(
    "CONTENT_API_URL",
    "http://10.56.184.54:8080/contentAPIURL"
)

CONTENT_API_TIMEOUT_SECONDS = int(os.getenv("CONTENT_API_TIMEOUT_SECONDS", "15"))

# GPT-5 family reasoning model.
# You can switch to gpt-5.5-pro, gpt-5.4, etc. through env vars.
MODEL_NAME = os.getenv(
    "BASELINE_REASONING_MODEL_NAME",
    os.getenv("REASONING_MODEL_NAME", "gpt-5.5")
)

REASONING_EFFORT = os.getenv(
    "BASELINE_REASONING_EFFORT",
    os.getenv("REASONING_EFFORT", "high")
)

# GPT-5.5 default pricing per 1M tokens.
# Override these if you switch to another GPT-5 family model.
INPUT_COST_PER_1M = float(
    os.getenv(
        "BASELINE_REASONING_INPUT_COST_PER_1M",
        os.getenv("REASONING_INPUT_COST_PER_1M", "5.00")
    )
)

OUTPUT_COST_PER_1M = float(
    os.getenv(
        "BASELINE_REASONING_OUTPUT_COST_PER_1M",
        os.getenv("REASONING_OUTPUT_COST_PER_1M", "30.00")
    )
)

# GPT-5.5 long-context pricing rule:
# prompts with >272K input tokens are priced at 2x input and 1.5x output
# for the full session.
LONG_CONTEXT_THRESHOLD_TOKENS = int(
    os.getenv("REASONING_LONG_CONTEXT_THRESHOLD_TOKENS", "272000")
)

LONG_CONTEXT_INPUT_MULTIPLIER = float(
    os.getenv("REASONING_LONG_CONTEXT_INPUT_MULTIPLIER", "2.0")
)

LONG_CONTEXT_OUTPUT_MULTIPLIER = float(
    os.getenv("REASONING_LONG_CONTEXT_OUTPUT_MULTIPLIER", "1.5")
)


# ------------------------------------------------------------------------------
# GENERIC OBJECT HELPERS
# ------------------------------------------------------------------------------

def read_field(obj, key, default=None):
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


# ------------------------------------------------------------------------------
# USAGE / COST HELPERS
# ------------------------------------------------------------------------------

def empty_usage():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "context_tokens": 0,
        "total_tokens": 0,
    }


def get_openai_usage(resp):
    usage = read_field(resp, "usage")

    if not usage:
        return empty_usage()

    input_tokens = (
        read_field(usage, "input_tokens")
        or read_field(usage, "prompt_tokens")
        or 0
    )

    output_tokens = (
        read_field(usage, "output_tokens")
        or read_field(usage, "completion_tokens")
        or 0
    )

    total_tokens = read_field(usage, "total_tokens")

    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    output_details = read_field(usage, "output_tokens_details", {}) or {}
    reasoning_tokens = read_field(output_details, "reasoning_tokens", 0) or 0

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        # For your evaluator, context_tokens means actual prompt/input tokens.
        "context_tokens": input_tokens,
        "total_tokens": total_tokens,
    }


def get_effective_pricing(usage):
    input_price = INPUT_COST_PER_1M
    output_price = OUTPUT_COST_PER_1M
    long_context_applied = False

    input_tokens = usage.get("input_tokens", 0) or 0

    if (
        MODEL_NAME.startswith("gpt-5.5")
        and input_tokens > LONG_CONTEXT_THRESHOLD_TOKENS
    ):
        input_price *= LONG_CONTEXT_INPUT_MULTIPLIER
        output_price *= LONG_CONTEXT_OUTPUT_MULTIPLIER
        long_context_applied = True

    return input_price, output_price, long_context_applied


def calculate_cost(usage):
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0

    input_price, output_price, long_context_applied = get_effective_pricing(usage)

    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost

    return {
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(total_cost, 8),
        "input_cost_per_1m_tokens": input_price,
        "output_cost_per_1m_tokens": output_price,
        "base_input_cost_per_1m_tokens": INPUT_COST_PER_1M,
        "base_output_cost_per_1m_tokens": OUTPUT_COST_PER_1M,
        "long_context_pricing_applied": long_context_applied,
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
        "reasoning_effort": REASONING_EFFORT,
    }), status_code


# ------------------------------------------------------------------------------
# RESPONSE PARSING HELPERS
# ------------------------------------------------------------------------------

def extract_response_text(resp):
    output_text = read_field(resp, "output_text")

    if output_text:
        return str(output_text).strip()

    output = read_field(resp, "output", []) or []
    chunks = []

    for item in output:
        item_type = read_field(item, "type")

        if item_type != "message":
            continue

        content = read_field(item, "content", []) or []

        for content_item in content:
            content_type = read_field(content_item, "type")

            if content_type in {"output_text", "text"}:
                text = read_field(content_item, "text", "")
                if text:
                    chunks.append(str(text))

    return "\n".join(chunks).strip()


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


def get_citation_snippet(item):
    snippet = "Location not specified"

    if isinstance(item.get("label"), dict) and item["label"].get("name"):
        snippet = str(item["label"].get("name")).strip()
    elif item.get("location"):
        snippet = str(item.get("location")).strip()
    elif item.get("host"):
        snippet = str(item.get("host")).strip()

    return snippet


# ------------------------------------------------------------------------------
# ROUTE
# ------------------------------------------------------------------------------

@ask_baseline_reasoning_bp.route("/ask_baseline_reasoning", methods=["POST"])
def ask_ai_reasoning_baseline():
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

        # Reasoning baseline intentionally sends full matching context.
        # No condensing. No truncation. No max_tokens request cap.
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

Your task is to answer the user's query by reasoning over the provided full JSON array of campus items.

USER QUERY:
{query}

AVAILABLE ITEMS:
{json.dumps(valid_items, indent=2, ensure_ascii=False, default=str)}

INSTRUCTIONS:
1. Use the full item context above.
2. Carefully identify which items directly answer the user's query.
3. Provide a comprehensive overview in 1-3 sentences.
4. Rank up to 10 relevant item IDs from best to worst match.
5. Only rank item IDs that appear in AVAILABLE ITEMS.
6. Output your ranked IDs EXACTLY in this format at the very end of your response:
[IDS: id1, id2, id3]
""".strip()

        resp = client.responses.create(
            model=MODEL_NAME,
            reasoning={
                "effort": REASONING_EFFORT
            },
            input=[
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ],
        )

        raw_output = extract_response_text(resp)

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
            "reasoning_effort": REASONING_EFFORT,
        }), 200

    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch content API data",
            "details": str(e),
        }), 502

    except Exception as e:
        return jsonify({
            "error": "Failed to generate reasoning AI response",
            "details": str(e),
        }), 500