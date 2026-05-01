# ask_enhanced.py

from flask import request, jsonify, Blueprint
import json
import os
import re
import requests
import math
from collections import Counter
import unicodedata
from difflib import SequenceMatcher

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ask_enhanced_bp = Blueprint("ask_enhanced", __name__)


# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------

CONTENT_API_URL = os.getenv("CONTENT_API_URL", "http://10.56.184.54:8080/contentAPIURL")
CONTENT_API_TIMEOUT_SECONDS = int(os.getenv("CONTENT_API_TIMEOUT_SECONDS", "15"))

MODEL_NAME = os.getenv("ENHANCED_OPENAI_MODEL_NAME", os.getenv("OPENAI_MODEL_NAME", "gpt-4o"))

# GPT-4o defaults. Override in .env if model/pricing changes.
# Current OpenAI docs list GPT-4o text input at $2.50 / 1M tokens
# and output at $10.00 / 1M tokens.
INPUT_COST_PER_1M = float(
    os.getenv("ENHANCED_INPUT_COST_PER_1M", os.getenv("OPENAI_INPUT_COST_PER_1M", "2.50"))
)
OUTPUT_COST_PER_1M = float(
    os.getenv("ENHANCED_OUTPUT_COST_PER_1M", os.getenv("OPENAI_OUTPUT_COST_PER_1M", "10.00"))
)

MAX_DESC_CHARS = int(os.getenv("ENHANCED_MAX_DESC_CHARS", "2000"))

# This is the main cost-control setting.
# Backup used 25. I set default to 15 so enhanced should be meaningfully cheaper.
# Change to 10 or 25 through .env if needed.
MAX_CONTEXT_ITEMS = int(os.getenv("ENHANCED_MAX_CONTEXT_ITEMS", "15"))
MIN_CONTEXT_ITEMS = int(os.getenv("ENHANCED_MIN_CONTEXT_ITEMS", "4"))

MAX_NESTED_SEARCH_CHARS = int(os.getenv("ENHANCED_MAX_NESTED_SEARCH_CHARS", "16000"))
MAX_NESTED_CONTEXT_CHARS = int(os.getenv("ENHANCED_MAX_NESTED_CONTEXT_CHARS", "1200"))
MAX_BLOB_FOR_FUZZY = int(os.getenv("ENHANCED_MAX_BLOB_FOR_FUZZY", "1800"))
MAX_SEGMENT_FOR_FUZZY = int(os.getenv("ENHANCED_MAX_SEGMENT_FOR_FUZZY", "500"))

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

SKIP_NESTED_KEYS = {
    "image_urls", "pin_url", "source_url", "url", "urls", "href", "link", "links"
}


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
        # This is now the real prompt/input token count.
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
        "context_config": {
            "max_context_items": MAX_CONTEXT_ITEMS,
            "min_context_items": MIN_CONTEXT_ITEMS,
            "max_nested_context_chars": MAX_NESTED_CONTEXT_CHARS,
        },
    }), status_code


# ------------------------------------------------------------------------------
# STOP WORDS & DYNAMIC WORD BANK
# ------------------------------------------------------------------------------

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "is", "are", "was", "were", "it", "this", "that", "these", "those",
    "then", "just", "so", "than", "such", "both", "through", "about", "while",
    "during", "what", "they", "we", "he", "she", "if", "because", "as", "when",
    "where", "how", "who", "which", "be", "has", "have", "had", "do", "does", "did"
}

DYNAMIC_EXPANSIONS = {}


# ------------------------------------------------------------------------------
# STATIC WORD BANK / ALIASES
# ------------------------------------------------------------------------------

DAY_ALIASES = {
    "monday": {"monday", "mon"},
    "tuesday": {"tuesday", "tue", "tues"},
    "wednesday": {"wednesday", "wed"},
    "thursday": {"thursday", "thu", "thur", "thurs"},
    "friday": {"friday", "fri"},
    "saturday": {"saturday", "sat"},
    "sunday": {"sunday", "sun"},
}

MEAL_ALIASES = {
    "breakfast": {"breakfast", "bf", "brkfst", "morning"},
    "lunch": {"lunch", "midday", "noon"},
    "dinner": {"dinner", "supper", "evening"},
    "bakery": {
        "bakery", "dessert", "pastry", "pastries", "cake",
        "cookies", "muffin", "croissant", "strudel"
    },
}

LOCATION_ALIASES = {
    "pavilion": {"pavilion", "pav"},
}

TOKEN_WORD_BANK = {
    "pav": {"pavilion"}, "pavilion": {"pav"},
    "fri": {"friday"}, "friday": {"fri"},
    "thu": {"thursday"}, "thurs": {"thursday"}, "thursday": {"thu", "thurs"},
    "wed": {"wednesday"}, "wednesday": {"wed"},
    "tue": {"tuesday"}, "tues": {"tuesday"}, "tuesday": {"tue", "tues"},
    "mon": {"monday"}, "monday": {"mon"},
    "sat": {"saturday"}, "saturday": {"sat"},
    "sun": {"sunday"}, "sunday": {"sun"},

    "veggie": {"vegetarian", "vegan", "plant", "plantbased"},
    "veg": {"vegetarian", "vegan", "plant", "plantbased"},
    "vegan": {"vegetarian", "plant", "plantbased"},
    "vegetarian": {"vegan", "plant", "plantbased"},
    "plant": {"plantbased", "vegan", "vegetarian"},
    "plantbased": {"plant", "vegan", "vegetarian"},
    "gf": {"gluten", "free", "glutenfree"},
    "glutenfree": {"gluten", "free", "gf"},
    "coffee": {"decaf", "drinks", "tea"},
    "tea": {"drinks", "coffee"},
    "burger": {"burgers"},
    "taco": {"tacos"},
    "ramen": {"noodle", "pho"},
    "pho": {"ramen", "noodle"},
    "pizza": {"pies"},
    "salad": {"greens"},

    "park": {"parking", "lot"},
    "parking": {"park", "lot"},
    "lot": {"park", "parking"},

    "crime": {"police", "ucpd", "safety", "security", "emergency", "report"},
    "report": {"police", "ucpd", "safety", "security", "emergency", "crime"},
    "reported": {"police", "ucpd", "safety", "security", "emergency", "crime", "report"},
    "reporting": {"police", "ucpd", "safety", "security", "emergency", "crime", "report"},
    "emergency": {"police", "ucpd", "safety", "security", "crime", "report"},
    "safety": {"police", "ucpd", "security", "emergency", "crime", "report"},
    "security": {"police", "ucpd", "safety", "emergency", "crime", "report"},
    "police": {"ucpd", "safety", "security", "emergency", "crime", "report"},
    "ucpd": {"police", "safety", "security", "emergency", "crime", "report"},
    "911": {"police", "ucpd", "safety", "security", "emergency", "crime"},
    "dispatch": {"police", "ucpd", "safety", "security", "emergency"},
    "law": {"police", "ucpd", "safety", "security"},
    "enforcement": {"police", "ucpd", "safety", "security"},
}

PHRASE_WORD_BANK = {
    "gluten free": {"glutenfree", "gf"},
    "plant based": {"plantbased", "vegan", "vegetarian"},

    "report a crime": {"police", "ucpd", "safety", "security", "emergency", "dispatch"},
    "report crime": {"police", "ucpd", "safety", "security", "emergency", "dispatch"},
    "campus police": {"police", "ucpd", "safety", "security", "emergency"},
    "campus safety": {"police", "ucpd", "safety", "security", "emergency"},
    "law enforcement": {"police", "ucpd", "safety", "security"},
}


# ------------------------------------------------------------------------------
# TEXT HELPERS
# ------------------------------------------------------------------------------

def make_singular(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("es") and not word.endswith("ss"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def normalize_text(value: str) -> str:
    if value is None:
        return ""

    text = str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s:/=\-|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    tokens = set()

    for tok in normalize_text(text).split():
        if len(tok) > 1 and tok not in STOP_WORDS:
            tokens.add(tok)
            tokens.add(make_singular(tok))

    return tokens


def strip_urls(text: str) -> str:
    if not text:
        return ""

    text = URL_RE.sub("", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_description(text: str, limit: int = MAX_DESC_CHARS) -> str:
    text = strip_urls(text)

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "..."


def compact_nested_value(value: str) -> str:
    text = strip_urls(value)

    if not text:
        return ""

    m = re.match(r"^\s*station\s*:\s*(.+)$", text, re.IGNORECASE)
    if m:
        return f"station={m.group(1).strip()}"

    m = re.match(r"^\s*description\s*:\s*(.+)$", text, re.IGNORECASE)
    if m:
        return f"desc={m.group(1).strip()}"

    m = re.match(r"^\s*calories\s*:\s*([0-9]+)", text, re.IGNORECASE)
    if m:
        return f"cal={m.group(1).strip()}"

    return text.strip(" |;,-")


def join_with_limit(parts, max_chars):
    out = []
    total = 0

    for part in parts:
        if not part:
            continue

        add_len = len(part) if not out else len(part) + 5

        if total + add_len > max_chars:
            break

        out.append(part)
        total += add_len

    return " ### ".join(out)


# ------------------------------------------------------------------------------
# DYNAMIC GENERATION HELPERS
# ------------------------------------------------------------------------------

def extract_plain_sections_text_for_dynamic_bank(item: dict) -> str:
    sections = item.get("sections", [])

    if not isinstance(sections, list):
        return ""

    parts = []

    for section in sections:
        if not isinstance(section, dict):
            continue

        header = section.get("header", "")
        body = section.get("body", "")
        bullets = section.get("bullets", [])

        if header:
            parts.append(str(header))

        if body:
            parts.append(str(body))

        if isinstance(bullets, list):
            parts.extend(str(b) for b in bullets if b)

    return " ".join(parts)


def generate_dynamic_word_bank(pages: list):
    global DYNAMIC_EXPANSIONS
    DYNAMIC_EXPANSIONS.clear()

    all_words = []

    for p in pages:
        sections_text = extract_plain_sections_text_for_dynamic_bank(p)

        text_blob = (
            f"{p.get('title', '')} "
            f"{p.get('description', '')} "
            f"{json.dumps(p.get('nested_content', ''))} "
            f"{sections_text}"
        )

        norm = normalize_text(text_blob)

        words = [
            make_singular(w)
            for w in norm.split()
            if len(w) > 2 and w not in STOP_WORDS
        ]

        all_words.extend(words)

    if not all_words:
        return

    counts = Counter(all_words)
    freqs = list(counts.values())

    mean_freq = sum(freqs) / len(freqs)
    variance = sum((f - mean_freq) ** 2 for f in freqs) / len(freqs)
    std_dev = math.sqrt(variance) if variance > 0 else 1

    valid_words = [
        w for w, f in counts.items()
        if (mean_freq - std_dev) <= f <= (mean_freq + std_dev)
    ]

    for w in valid_words:
        sing = make_singular(w)

        DYNAMIC_EXPANSIONS.setdefault(sing, set()).add(w)

        if w != sing:
            DYNAMIC_EXPANSIONS.setdefault(w, set()).add(sing)


# ------------------------------------------------------------------------------
# QUERY / WORD BANK HELPERS
# ------------------------------------------------------------------------------

def detect_canonical_matches(text: str, alias_map: dict) -> set:
    norm_text = normalize_text(text)
    tokens = set(norm_text.split())
    found = set()

    for canonical, aliases in alias_map.items():
        variants = set(aliases) | {canonical}

        for alias in variants:
            alias_norm = normalize_text(alias)

            if not alias_norm:
                continue

            if " " in alias_norm:
                if alias_norm in norm_text:
                    found.add(canonical)
                    break
            else:
                if alias_norm in tokens or make_singular(alias_norm) in tokens:
                    found.add(canonical)
                    break

    return found


def build_query_hints(query: str) -> dict:
    query_norm = normalize_text(query)
    raw_tokens = set(query_norm.split())
    expanded_tokens = set(raw_tokens)

    for tok in list(raw_tokens):
        expanded_tokens.add(make_singular(tok))

    for phrase, expansions in PHRASE_WORD_BANK.items():
        phrase_norm = normalize_text(phrase)

        if phrase_norm in query_norm:
            expanded_tokens.update(expansions)

    for tok in list(expanded_tokens):
        expanded_tokens.update(TOKEN_WORD_BANK.get(tok, set()))
        expanded_tokens.update(DYNAMIC_EXPANSIONS.get(tok, set()))

    days = detect_canonical_matches(query, DAY_ALIASES)
    meals = detect_canonical_matches(query, MEAL_ALIASES)
    locations = detect_canonical_matches(query, LOCATION_ALIASES)

    for d in days:
        expanded_tokens.add(d)
        expanded_tokens.update(DAY_ALIASES.get(d, set()))

    for m in meals:
        expanded_tokens.add(m)
        expanded_tokens.update(MEAL_ALIASES.get(m, set()))

    for loc in locations:
        expanded_tokens.add(loc)
        expanded_tokens.update(LOCATION_ALIASES.get(loc, set()))

    return {
        "query": query,
        "query_norm": query_norm,
        "raw_tokens": raw_tokens,
        "expanded_tokens": expanded_tokens,
        "days": days,
        "meals": meals,
        "locations": locations,
    }


def canonicalize_from_aliases(value: str, alias_map: dict) -> str:
    matches = detect_canonical_matches(value or "", alias_map)

    if matches:
        return sorted(matches)[0]

    return normalize_text(value or "")


# ------------------------------------------------------------------------------
# NESTED CONTENT EXTRACTION
# ------------------------------------------------------------------------------

def gather_generic_strings(obj, fragments):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in SKIP_NESTED_KEYS:
                continue

            if isinstance(value, str):
                compact = compact_nested_value(value)

                if compact:
                    fragments.append(compact)

            elif isinstance(value, (dict, list)):
                gather_generic_strings(value, fragments)

    elif isinstance(obj, list):
        for item in obj:
            gather_generic_strings(item, fragments)

    elif isinstance(obj, str):
        compact = compact_nested_value(obj)

        if compact:
            fragments.append(compact)


def make_segment_text(day_title: str, tab_title: str, header: str, extra_parts: list) -> str:
    parts = []

    if day_title:
        parts.append(f"day={day_title}")

    if tab_title:
        parts.append(f"tab={tab_title}")

    if header:
        parts.append(f"item={header}")

    for part in extra_parts:
        part = compact_nested_value(part)

        if part:
            parts.append(part)

    deduped = []
    seen = set()

    for part in parts:
        norm = normalize_text(part)

        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(part)

    return " || ".join(deduped).strip()


def make_segment_dict(
    idx: int,
    day_title: str,
    tab_title: str,
    header: str,
    text: str
) -> dict:
    return {
        "idx": idx,
        "day": day_title,
        "tab": tab_title,
        "header": header,
        "text": text,
        "normalized": normalize_text(text),
        "token_set": tokenize(text),
        "day_canonical": canonicalize_from_aliases(day_title, DAY_ALIASES),
        "tab_canonical": canonicalize_from_aliases(tab_title, MEAL_ALIASES),
    }


def extract_nested_segments(nested_content, start_idx=0):
    segments = []
    seq = start_idx

    if not nested_content:
        return segments

    if isinstance(nested_content, list):
        for day_obj in nested_content:
            if not isinstance(day_obj, dict):
                continue

            day_title = strip_urls(day_obj.get("title", "") or "")
            tabs = day_obj.get("tabs", [])

            if isinstance(tabs, list) and tabs:
                for tab_obj in tabs:
                    if not isinstance(tab_obj, dict):
                        continue

                    tab_title = strip_urls(tab_obj.get("title", "") or "")
                    sections = tab_obj.get("sections", [])

                    if isinstance(sections, list) and sections:
                        for section in sections:
                            if not isinstance(section, dict):
                                continue

                            header = strip_urls(section.get("header", "") or "")
                            extra_parts = []

                            bullets = section.get("bullets", [])

                            if isinstance(bullets, list):
                                extra_parts.extend(bullets)

                            for key, value in section.items():
                                if key in {"header", "bullets"} or key in SKIP_NESTED_KEYS:
                                    continue

                                if isinstance(value, str):
                                    extra_parts.append(f"{key}={value}")

                            text = make_segment_text(day_title, tab_title, header, extra_parts)

                            if not text:
                                continue

                            segments.append(
                                make_segment_dict(
                                    idx=seq,
                                    day_title=day_title,
                                    tab_title=tab_title,
                                    header=header,
                                    text=text
                                )
                            )

                            seq += 1

                    else:
                        fallback_parts = []
                        gather_generic_strings(tab_obj, fallback_parts)

                        text = make_segment_text(day_title, tab_title, "", fallback_parts)

                        if text:
                            segments.append(
                                make_segment_dict(
                                    idx=seq,
                                    day_title=day_title,
                                    tab_title=tab_title,
                                    header="",
                                    text=text
                                )
                            )

                            seq += 1

            else:
                fallback_parts = []
                gather_generic_strings(day_obj, fallback_parts)

                text = make_segment_text(day_title, "", "", fallback_parts)

                if text:
                    segments.append(
                        make_segment_dict(
                            idx=seq,
                            day_title=day_title,
                            tab_title="",
                            header="",
                            text=text
                        )
                    )

                    seq += 1

    else:
        fallback_parts = []
        gather_generic_strings(nested_content, fallback_parts)

        text = make_segment_text("", "", "", fallback_parts)

        if text:
            segments.append(
                make_segment_dict(
                    idx=seq,
                    day_title="",
                    tab_title="",
                    header="",
                    text=text
                )
            )

    return segments


def extract_section_segments(sections, start_idx=0):
    segments = []
    seq = start_idx

    if not isinstance(sections, list):
        return segments

    for section in sections:
        if not isinstance(section, dict):
            continue

        header = strip_urls(section.get("header", "") or "")
        extra_parts = []

        body = section.get("body", "")
        bullets = section.get("bullets", [])

        if body:
            extra_parts.append(f"body={body}")

        if isinstance(bullets, list):
            extra_parts.extend(str(b) for b in bullets if b)

        for key, value in section.items():
            if key in {"header", "body", "bullets"} or key in SKIP_NESTED_KEYS:
                continue

            if isinstance(value, str):
                extra_parts.append(f"{key}={value}")
            elif isinstance(value, (dict, list)):
                fallback_parts = []
                gather_generic_strings(value, fallback_parts)
                extra_parts.extend(fallback_parts)

        text = make_segment_text("", "", header, extra_parts)

        if not text:
            continue

        segments.append(
            make_segment_dict(
                idx=seq,
                day_title="",
                tab_title="",
                header=header,
                text=text
            )
        )

        seq += 1

    return segments


def build_sections_for_llm(sections):
    if not isinstance(sections, list):
        return []

    cleaned_sections = []

    for section in sections:
        if not isinstance(section, dict):
            continue

        cleaned = {}

        header = section.get("header", "")
        body = section.get("body", "")
        bullets = section.get("bullets", [])

        if header:
            cleaned["header"] = str(header)

        if body:
            cleaned["body"] = str(body)

        if isinstance(bullets, list) and bullets:
            cleaned["bullets"] = [str(b) for b in bullets if b]

        for key, value in section.items():
            if key in {"header", "body", "bullets"} or key in SKIP_NESTED_KEYS:
                continue

            if isinstance(value, (str, int, float, bool)) and value is not None:
                cleaned[key] = value

        if cleaned:
            cleaned_sections.append(cleaned)

    return cleaned_sections


def collapse_structured_segments(segments, max_chars=MAX_NESTED_SEARCH_CHARS) -> str:
    return join_with_limit([seg["text"] for seg in segments], max_chars)


# ------------------------------------------------------------------------------
# SEGMENT SCORING / QUERY-AWARE EXCERPT
# ------------------------------------------------------------------------------

def score_segment(query_hints: dict, segment: dict) -> float:
    expanded_tokens = query_hints["expanded_tokens"]
    query_norm = query_hints["query_norm"]

    if not query_norm:
        return 0.0

    seg_norm = segment["normalized"][:MAX_SEGMENT_FOR_FUZZY]
    seg_tokens = segment["token_set"]

    overlap = len(expanded_tokens & seg_tokens)
    overlap_ratio = overlap / max(len(expanded_tokens), 1)

    seq_ratio = SequenceMatcher(None, query_norm, seg_norm).ratio()
    contains_boost = 1.0 if query_norm in seg_norm else 0.0

    day_bonus = 0.0
    meal_bonus = 0.0

    if query_hints["days"] and segment["day_canonical"] in query_hints["days"]:
        day_bonus += 0.60

    if query_hints["meals"] and segment["tab_canonical"] in query_hints["meals"]:
        meal_bonus += 0.50

    header_hits = 0
    header_norm = normalize_text(segment.get("header", ""))

    if header_norm:
        header_hits = sum(1 for tok in expanded_tokens if tok in header_norm)

    score = (
        (overlap_ratio * 0.35) +
        (seq_ratio * 0.25) +
        (contains_boost * 0.15) +
        day_bonus +
        meal_bonus +
        (header_hits * 0.08)
    )

    return round(score, 6)


def build_query_aware_nested_excerpt(segments, query_hints, max_chars=MAX_NESTED_CONTEXT_CHARS) -> str:
    if not segments:
        return ""

    scored = []

    for seg in segments:
        seg_score = score_segment(query_hints, seg)
        scored.append((seg_score, seg["idx"], seg))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    selected = []
    selected_chars = 0
    group_counts = {}

    for seg_score, _, seg in scored:
        if seg_score <= 0 and selected:
            continue

        group_key = (seg.get("day_canonical", ""), seg.get("tab_canonical", ""))

        if group_counts.get(group_key, 0) >= 4:
            continue

        text = seg["text"]
        add_len = len(text) if not selected else len(text) + 5

        if selected_chars + add_len > max_chars:
            continue

        selected.append(seg)
        selected_chars += add_len
        group_counts[group_key] = group_counts.get(group_key, 0) + 1

        if len(selected) >= 8:
            break

    if not selected:
        for seg in sorted(segments, key=lambda s: s["idx"]):
            text = seg["text"]
            add_len = len(text) if not selected else len(text) + 5

            if selected_chars + add_len > max_chars:
                break

            selected.append(seg)
            selected_chars += add_len

    selected.sort(key=lambda s: s["idx"])

    return " ### ".join(seg["text"] for seg in selected)


# ------------------------------------------------------------------------------
# ITEM ENCODING / SCORING
# ------------------------------------------------------------------------------

def encode_item(item: dict) -> dict:
    title = item.get("title", "") or ""
    subtitle = item.get("subtitle", "") or ""
    host = item.get("host", "") or ""
    description = item.get("description", "") or ""

    tags = item.get("tags", []) or []
    item_type = item.get("type", "") or ""
    start = item.get("start", "") or ""
    end = item.get("end", "") or ""

    nested_content_segments = extract_nested_segments(
        item.get("nested_content", []),
        start_idx=0
    )

    section_segments = extract_section_segments(
        item.get("sections", []),
        start_idx=len(nested_content_segments)
    )

    all_segments = nested_content_segments + section_segments

    structured_text = collapse_structured_segments(
        all_segments,
        max_chars=MAX_NESTED_SEARCH_CHARS
    )

    tags_text = " ".join(str(t) for t in tags)

    search_blob = " | ".join(
        part
        for part in [
            title,
            subtitle,
            host,
            tags_text,
            description,
            structured_text,
            item_type,
        ]
        if part
    )

    compact_item = {
        "id": item.get("id"),
        "title": title,
        "subtitle": subtitle,
        "host": host,
        "description": compact_description(description),
        "tags": tags,
        "type": item_type,
        "start": start,
        "end": end,
    }

    llm_sections = build_sections_for_llm(item.get("sections", []))

    if llm_sections:
        compact_item["sections"] = llm_sections

    return {
        "raw": item,
        "compact": compact_item,
        "search_blob": search_blob,
        "normalized_blob": normalize_text(search_blob),
        "token_set": tokenize(search_blob),
        "nested_segments": all_segments,
        "nested_structured_text": structured_text,
    }


def score_encoded_item(query_hints: dict, encoded: dict) -> float:
    query_norm = query_hints["query_norm"]
    expanded_tokens = query_hints["expanded_tokens"]

    if not query_norm:
        return 0.0

    compact = encoded["compact"]
    blob = encoded["normalized_blob"][:MAX_BLOB_FOR_FUZZY]
    token_set = encoded["token_set"]

    title_norm = normalize_text(compact.get("title", ""))
    subtitle_norm = normalize_text(compact.get("subtitle", ""))
    host_norm = normalize_text(compact.get("host", ""))
    desc_norm = normalize_text(compact.get("description", ""))

    title_token_set = tokenize(compact.get("title", ""))
    subtitle_token_set = tokenize(compact.get("subtitle", ""))
    host_token_set = tokenize(compact.get("host", ""))
    desc_token_set = tokenize(compact.get("description", ""))
    tag_token_set = tokenize(" ".join(compact.get("tags", [])))

    overlap = len(expanded_tokens & token_set)
    overlap_ratio = overlap / max(len(expanded_tokens), 1)

    seq_ratio = SequenceMatcher(None, query_norm, blob).ratio()
    contains_boost = 1.0 if query_norm in blob else 0.0

    title_hits = len(expanded_tokens & title_token_set)
    subtitle_hits = len(expanded_tokens & subtitle_token_set)
    host_hits = len(expanded_tokens & host_token_set)
    desc_hits = len(expanded_tokens & desc_token_set)
    tag_hits = len(expanded_tokens & tag_token_set)

    if title_norm:
        title_hits += sum(
            1 for tok in expanded_tokens
            if len(tok) > 3 and tok in title_norm and tok not in title_token_set
        )

    if subtitle_norm:
        subtitle_hits += sum(
            1 for tok in expanded_tokens
            if len(tok) > 3 and tok in subtitle_norm and tok not in subtitle_token_set
        )

    if host_norm:
        host_hits += sum(
            1 for tok in expanded_tokens
            if len(tok) > 3 and tok in host_norm and tok not in host_token_set
        )

    if desc_norm:
        desc_hits += sum(
            1 for tok in expanded_tokens
            if len(tok) > 3 and tok in desc_norm and tok not in desc_token_set
        )

    field_boost = (
        (title_hits * 0.16) +
        (subtitle_hits * 0.08) +
        (host_hits * 0.06) +
        (desc_hits * 0.18) +
        (tag_hits * 0.20)
    )

    nested_scores = [
        score_segment(query_hints, seg)
        for seg in encoded["nested_segments"]
    ]

    best_nested_score = max(nested_scores) if nested_scores else 0.0
    nested_match_count = sum(1 for s in nested_scores if s >= 0.40)

    score = (
        (overlap_ratio * 0.30) +
        (seq_ratio * 0.20) +
        (contains_boost * 0.12) +
        field_boost +
        (best_nested_score * 0.40) +
        (min(nested_match_count, 3) * 0.07)
    )

    return round(score, 6)


# ------------------------------------------------------------------------------
# RESPONSE HELPERS
# ------------------------------------------------------------------------------

def parse_ranked_ids(raw_output, valid_item_ids):
    ranked_item_ids = []

    tag_match = re.search(r"\[IDS:\s*(.*?)\]", raw_output, re.IGNORECASE | re.DOTALL)

    if tag_match:
        found_ids = [
            x.strip()
            for x in tag_match.group(1).split(",")
            if x.strip()
        ]

        ranked_item_ids = [
            x for x in found_ids
            if x in valid_item_ids
        ]

    else:
        fallback_match = re.search(r"IDs?:\s*(.*)", raw_output, re.IGNORECASE)

        if fallback_match:
            found_ids = fallback_match.group(1).replace(",", " ").split()

            ranked_item_ids = [
                x.strip()
                for x in found_ids
                if x.strip() in valid_item_ids
            ]

    return ranked_item_ids


def clean_ai_overview(raw_output):
    ai_overview = re.sub(
        r"\[IDS:.*?\]",
        "",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL
    ).strip()

    ai_overview = re.sub(
        r"(?i)IDs?:.*",
        "",
        ai_overview
    ).strip()

    if not ai_overview:
        ai_overview = "Here are the top matches based on your search."

    return ai_overview


def get_citation_snippet(item):
    snippet = ""

    if isinstance(item.get("label"), dict) and item["label"].get("name"):
        snippet = str(item["label"].get("name")).strip()
    elif item.get("location"):
        snippet = str(item.get("location")).strip()
    elif item.get("host"):
        snippet = str(item.get("host")).strip()

    if not snippet:
        snippet = "Location not specified"

    return snippet


# ------------------------------------------------------------------------------
# AI SMART SEARCH ROUTE
# ------------------------------------------------------------------------------

@ask_enhanced_bp.route("/ai_enhanced", methods=["POST"])
def ask_ai():
    data = request.get_json(silent=True) or {}

    query = str(data.get("query", "")).strip()
    item_ids = data.get("item_ids", [])

    try:
        if not query:
            return jsonify({"error": "No query provided"}), 400

        if not isinstance(item_ids, list):
            return jsonify({"error": "item_ids must be an array"}), 400

        if not item_ids:
            return build_empty_response("No item_ids were provided in the request.")

        content_resp = requests.get(
            CONTENT_API_URL,
            timeout=CONTENT_API_TIMEOUT_SECONDS
        )
        content_resp.raise_for_status()

        content_json = content_resp.json()
        pages = content_json.get("pages", [])

        if not isinstance(pages, list):
            return jsonify({"error": "Invalid content API response"}), 500

        generate_dynamic_word_bank(pages)

        query_hints = build_query_hints(query)

        item_id_set = set(item_ids)

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

        encoded_items = [
            encode_item(item)
            for item in valid_items
        ]

        scored = []

        for enc in encoded_items:
            local_score = score_encoded_item(query_hints, enc)

            scored.append({
                "score": local_score,
                "encoded": enc,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        # This preserves the backup behavior: rank all valid items locally,
        # but only send the top N compact candidates into the LLM.
        top_scored = scored[:MAX_CONTEXT_ITEMS]

        if len(top_scored) < MIN_CONTEXT_ITEMS and len(scored) >= MIN_CONTEXT_ITEMS:
            top_scored = scored[:MIN_CONTEXT_ITEMS]
        elif len(top_scored) == 0 and scored:
            top_scored = scored

        llm_candidates = []

        for row in top_scored:
            enc = row["encoded"]
            compact = dict(enc["compact"])
            compact["_retrieval_score"] = row["score"]

            nested_compact = build_query_aware_nested_excerpt(
                enc["nested_segments"],
                query_hints,
                max_chars=MAX_NESTED_CONTEXT_CHARS,
            )

            if nested_compact:
                compact["nested_content_compact"] = nested_compact

            llm_candidates.append(compact)

        system_prompt = (
            "You are a campus informant. "
            "Answer the user's query using the provided campus items. "
            "The items were selected by a local retrieval/ranking layer and are ordered by relevance. "
            "Provide a comprehensive overview, at least 1-3 sentences, summarizing the relevant items. "
            "Only recommend items that are directly relevant to the user's query. "
            "If one item clearly answers the query better than the others, rank that item first. "
            "Include up to 10 relevant item IDs in your ranking at the end, ranked best to worst in this exact format:\n"
            "[IDS: id1, id2, id3, id4, ...]\n"
            "Do not output JSON."
        )

        user_content = json.dumps({
            "query": query,
            "available_items": llm_candidates,
        }, indent=2, ensure_ascii=False, default=str)

        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=llm_messages,
            temperature=0.3,
        )

        raw_output = (resp.choices[0].message.content or "").strip()

        usage = get_openai_usage(resp)
        cost = calculate_cost(usage)

        ranked_item_ids = parse_ranked_ids(raw_output, valid_item_ids)

        if not ranked_item_ids:
            ranked_item_ids = [
                row["encoded"]["compact"]["id"]
                for row in top_scored
                if row["encoded"]["compact"].get("id")
            ]

        ai_overview = clean_ai_overview(raw_output)

        citations = []

        for pid in ranked_item_ids:
            matched_item = next(
                (item for item in valid_items if item.get("id") == pid),
                None
            )

            if not matched_item:
                continue

            citations.append({
                "page_id": pid,
                "title": matched_item.get("title", ""),
                "snippet": get_citation_snippet(matched_item),
            })

        final_response = {
            "ai_overview": ai_overview,
            "citations": citations,
            "ranked_item_ids": ranked_item_ids,
            "usage": usage,
            "cost": cost,
            "estimated_cost_usd": cost["total_cost_usd"],
            "model": MODEL_NAME,
            "context_config": {
                "candidate_item_count": len(valid_items),
                "ranked_item_count": len(scored),
                "llm_context_item_count": len(llm_candidates),
                "max_context_items": MAX_CONTEXT_ITEMS,
                "min_context_items": MIN_CONTEXT_ITEMS,
                "max_nested_context_chars": MAX_NESTED_CONTEXT_CHARS,
            },
        }

        return jsonify(final_response), 200

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