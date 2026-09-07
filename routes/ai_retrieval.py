from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html import unescape
from typing import Any

MAX_DESC_CHARS = 220
MAX_DESCRIPTION_CONTEXT_CHARS = 900
MAX_DESCRIPTION_EXCERPT_CHARS = 520
MAX_EVIDENCE_EXCERPT_CHARS = 900
MAX_CONTEXT_ITEMS = 30
MIN_CONTEXT_ITEMS = 4
MAX_NESTED_SEARCH_CHARS = 16000
MAX_NESTED_CONTEXT_CHARS = 1200
MAX_NESTED_EVIDENCE_CHARS = 520
MAX_SEGMENT_FOR_FUZZY = 500
MAX_CONTEXT_DUPLICATES_PER_KEY = 3

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

SKIP_NESTED_KEYS = {
    "image_urls",
    "pin_url",
    "source_url",
    "url",
    "urls",
    "href",
    "link",
    "links",
}

STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "is",
    "are",
    "was",
    "were",
    "it",
    "this",
    "that",
    "these",
    "those",
    "then",
    "just",
    "so",
    "than",
    "such",
    "both",
    "through",
    "about",
    "while",
    "during",
    "what",
    "they",
    "we",
    "he",
    "she",
    "if",
    "because",
    "as",
    "when",
    "where",
    "how",
    "who",
    "which",
    "be",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "dont",
    "doesnt",
    "not",
    "can",
    "could",
    "would",
    "should",
    "i",
    "me",
    "my",
    "our",
    "you",
    "your",
    "go",
    "get",
    "use",
    "need",
    "want",
    "looking",
    "find",
    "show",
    "tell",
    "any",
    "anywhere",
    "anything",
    "thing",
    "things",
    "will",
    "going",
    "out",
    "campus",
    "uc",
    "merced",
    "please",
    "near",
    "nearby",
    "closest",
    "around",
}

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
        "bakery",
        "dessert",
        "pastry",
        "pastries",
        "cake",
        "cookies",
        "muffin",
        "croissant",
        "strudel",
    },
}

LOCATION_ALIASES = {
    "pavilion": {"pavilion", "pav"},
    "library": {"library", "kolligian", "leo dottie"},
}

CONCEPT_ALIASES = {
    "restroom": {
        "restroom",
        "restrooms",
        "bathroom",
        "bathrooms",
        "toilet",
        "toilets",
        "washroom",
        "lavatory",
        "public restroom",
        "public bathroom",
    },
    "dining": {
        "dining",
        "food",
        "foods",
        "snack",
        "snacks",
        "refreshment",
        "refreshments",
        "drink",
        "drinks",
        "coffee",
        "donut",
        "donuts",
        "brunch",
        "eat",
        "eating",
        "meal",
        "meals",
        "restaurant",
        "restaurants",
        "cafeteria",
        "cafe",
        "pavilion",
        "ywdc",
        "lunch",
        "dinner",
        "breakfast",
    },
    "snacks": {
        "snack",
        "snacks",
        "vending",
        "vending machine",
        "coffee",
        "tea",
        "drink",
        "drinks",
        "donut",
        "donuts",
        "boba",
    },
    "parking": {
        "parking",
        "park",
        "lot",
        "lots",
        "permit",
        "permits",
        "taps",
        "transportation parking services",
        "ev charging",
    },
    "event": {
        "event",
        "events",
        "activity",
        "activities",
        "happening",
        "happenings",
        "tonight",
        "today",
        "upcoming",
        "fun",
        "game",
        "games",
        "workshop",
        "meeting",
        "social",
        "tabling",
    },
    "wildlife": {
        "wildlife",
        "sighting",
        "sightings",
        "animal",
        "animals",
        "bird",
        "birds",
        "duck",
        "ducks",
        "bunny",
        "rabbit",
        "rabbits",
        "snake",
        "deer",
        "insect",
        "insects",
    },
    "building": {
        "building",
        "buildings",
        "location",
        "locations",
        "place",
        "places",
        "hall",
        "room",
        "classroom",
        "office",
        "center",
    },
    "library": {
        "library",
        "kolligian library",
        "leo dottie kolligian",
    },
    "retail": {
        "retail",
        "store",
        "stores",
        "shop",
        "shops",
        "market",
        "bookstore",
    },
    "transit": {
        "transit",
        "transportation",
        "bus",
        "buses",
        "shuttle",
        "shuttles",
        "cattracks",
    },
    "research": {
        "research",
        "lab",
        "labs",
        "laboratory",
        "laboratories",
        "science",
        "engineering",
    },
}

CONCEPT_QUERY_EXPANSIONS = {
    "restroom": {"restroom", "bathroom", "toilet", "washroom", "lavatory"},
    "dining": {
        "dining",
        "food",
        "eat",
        "meal",
        "restaurant",
        "cafeteria",
        "cafe",
        "snack",
        "snacks",
        "refreshment",
        "refreshments",
        "drink",
        "drinks",
        "coffee",
        "donut",
        "donuts",
        "brunch",
    },
    "snacks": {
        "snack",
        "snacks",
        "vending",
        "coffee",
        "tea",
        "drink",
        "drinks",
        "refreshment",
        "refreshments",
        "donut",
        "donuts",
    },
    "parking": {"parking", "park", "lot", "permit", "taps"},
    "event": {"event", "events", "activity", "activities", "happening", "upcoming"},
    "wildlife": {"wildlife", "sighting", "animal", "bird", "duck", "rabbit"},
    "building": {"building", "buildings", "location", "place", "hall", "room", "office", "center"},
    "library": {"library", "kolligian"},
    "retail": {"retail", "store", "shop", "market", "bookstore"},
    "transit": {"transit", "transportation", "bus", "shuttle", "cattracks"},
    "research": {"research", "lab", "laboratory", "science", "engineering"},
}

TOKEN_WORD_BANK = {
    "pav": {"pavilion"},
    "pavilion": {"pav"},
    "fri": {"friday"},
    "friday": {"fri"},
    "thu": {"thursday"},
    "thurs": {"thursday"},
    "thursday": {"thu", "thurs"},
    "wed": {"wednesday"},
    "wednesday": {"wed"},
    "tue": {"tuesday"},
    "tues": {"tuesday"},
    "tuesday": {"tue", "tues"},
    "mon": {"monday"},
    "monday": {"mon"},
    "sat": {"saturday"},
    "saturday": {"sat"},
    "sun": {"sunday"},
    "sunday": {"sun"},
    "veggie": {"vegetarian", "vegan", "plant", "plantbased"},
    "veg": {"vegetarian", "vegan", "plant", "plantbased"},
    "vegan": {"vegetarian", "plant", "plantbased"},
    "vegetarian": {"vegan", "plant", "plantbased"},
    "plant": {"plantbased", "vegan", "vegetarian"},
    "plantbased": {"plant", "vegan", "vegetarian"},
    "gf": {"gluten", "free", "glutenfree"},
    "glutenfree": {"gluten", "free", "gf"},
    "food": {
        "meal",
        "dining",
        "snack",
        "snacks",
        "refreshment",
        "refreshments",
        "drink",
        "drinks",
        "coffee",
        "donut",
        "donuts",
        "brunch",
        "lunch",
        "dinner",
        "breakfast",
    },
    "foods": {"food", "meal", "dining", "snacks", "refreshments", "drinks"},
    "coffee": {"decaf", "drinks", "drink", "tea", "snacks", "food"},
    "tea": {"drinks", "drink", "coffee", "refreshments"},
    "drink": {"drinks", "coffee", "tea", "refreshment", "food"},
    "drinks": {"drink", "coffee", "tea", "refreshments", "food"},
    "refreshment": {"refreshments", "drink", "snack", "food"},
    "refreshments": {"refreshment", "drinks", "snacks", "food"},
    "donut": {"donuts", "coffee", "snack", "food"},
    "donuts": {"donut", "coffee", "snacks", "food"},
    "brunch": {"breakfast", "lunch", "food", "meal"},
    "snack": {"snacks", "vending", "food"},
    "snacks": {"snack", "vending", "food"},
    "fun": {"activity", "activities", "game", "games", "movie", "social"},
    "game": {"games", "fun", "activity"},
    "games": {"game", "fun", "activity"},
    "movie": {"film", "fun", "activity"},
    "burger": {"burgers"},
    "taco": {"tacos"},
    "ramen": {"noodle", "pho"},
    "pho": {"ramen", "noodle"},
    "pizza": {"pies"},
    "salad": {"greens"},
    "park": {"parking", "lot"},
    "parking": {"park", "lot"},
    "lot": {"park", "parking"},
    "free": {"complimentary", "included"},
    "complimentary": {"free", "included"},
    "offering": {"offer", "offers", "provide", "provides", "provided", "serving", "served", "giving", "give"},
    "offer": {"offering", "offers", "provide", "provided", "serving"},
    "offers": {"offer", "offering", "provide", "provided", "serving"},
    "provide": {"provided", "provides", "offering", "offer", "serving"},
    "provided": {"provide", "provides", "offering", "offer", "included"},
    "provides": {"provide", "provided", "offering", "offer"},
    "giving": {"give", "offering", "provided"},
    "give": {"giving", "offering", "provided"},
    "bathroom": {"restroom", "restrooms", "toilet"},
    "bathrooms": {"restroom", "restrooms", "toilet"},
    "restroom": {"bathroom", "toilet", "restrooms"},
    "restrooms": {"restroom", "bathroom", "toilet"},
    "toilet": {"restroom", "bathroom"},
    "bird": {"birds", "duck", "ducks", "wildlife"},
    "birds": {"bird", "duck", "ducks", "wildlife"},
    "duck": {"ducks", "bird", "birds", "wildlife"},
    "ducks": {"duck", "bird", "birds", "wildlife"},
}

PHRASE_WORD_BANK = {
    "gluten free": {"glutenfree", "gf"},
    "plant based": {"plantbased", "vegan", "vegetarian"},
    "go to the restroom": {"restroom", "bathroom"},
    "use the bathroom": {"bathroom", "restroom"},
    "go to the bathroom": {"bathroom", "restroom"},
    "grab food": {"food", "dining", "snacks"},
    "get food": {"food", "dining"},
    "offering food": {"offer", "offering", "provide", "provided", "food"},
    "giving out food": {"give", "giving", "provided", "food"},
    "food provided": {"food", "provide", "provided"},
    "food will be provided": {"food", "provide", "provided"},
    "coffee and snacks": {"coffee", "snacks", "refreshments"},
    "donuts and coffee": {"donuts", "coffee", "snacks"},
    "free parking": {"free", "parking", "park"},
    "park for free": {"free", "parking", "park"},
    "parking free": {"free", "parking", "park"},
    "doesnt cost money": {"free", "parking"},
    "does not cost money": {"free", "parking"},
    "no cost": {"free"},
    "going on": {"event", "events", "happening"},
}

QUALIFIER_ALIASES = {
    "free": {
        "free",
        "free of charge",
        "no cost",
        "without cost",
        "no charge",
        "doesnt cost",
        "doesnt cost money",
        "does not cost",
        "does not cost money",
        "costs nothing",
        "complimentary",
    },
    "provided": {
        "offering",
        "offer",
        "offers",
        "provide",
        "provides",
        "provided",
        "serving",
        "served",
        "giving",
        "give",
        "giving out",
    },
    "late": {"late", "late night", "open late", "night"},
    "open": {"open", "available", "hours"},
    "public": {"public"},
    "accessible": {"accessible", "accessibility", "ada", "wheelchair"},
    "indoor": {"indoor", "inside"},
    "outdoor": {"outdoor", "outside"},
    "quiet": {"quiet", "calm"},
}

QUALIFIER_MATCH_ALIASES = {
    "free": {
        "free",
        "free of charge",
        "no cost",
        "without cost",
        "no charge",
        "complimentary",
    },
    "provided": {
        "offering",
        "offer",
        "offers",
        "provide",
        "provides",
        "provided",
        "serving",
        "served",
        "giving",
        "give",
        "giving out",
    },
    "late": QUALIFIER_ALIASES["late"],
    "open": QUALIFIER_ALIASES["open"],
    "public": QUALIFIER_ALIASES["public"],
    "accessible": QUALIFIER_ALIASES["accessible"],
    "indoor": QUALIFIER_ALIASES["indoor"],
    "outdoor": QUALIFIER_ALIASES["outdoor"],
    "quiet": QUALIFIER_ALIASES["quiet"],
}

QUALIFIER_QUERY_EXPANSIONS = {
    "free": {"free", "complimentary"},
    "provided": {"offer", "offering", "provide", "provided", "serving", "giving"},
    "late": {"late", "night"},
    "open": {"open", "available", "hours"},
    "public": {"public"},
    "accessible": {"accessible", "accessibility", "ada", "wheelchair"},
    "indoor": {"indoor", "inside"},
    "outdoor": {"outdoor", "outside"},
    "quiet": {"quiet", "calm"},
}

PROXIMITY_ALIASES = {
    "near me",
    "nearby",
    "closest",
    "nearest",
    "around me",
}

FIELD_WEIGHTS = {
    "title": 3.1,
    "type": 2.8,
    "tags": 2.3,
    "label": 2.0,
    "subtitle": 1.3,
    "location": 1.2,
    "host": 0.6,
    "description": 0.8,
    "nested": 0.55,
}

SIGNAL_WEIGHTS = {
    "bm25": 0.72,
    "exact_title": 2.5,
    "all_title_terms": 1.2,
    "title_token": 0.9,
    "structured_concept": 2.4,
    "title_concept": 1.3,
    "tag_concept": 1.0,
    "location_token": 0.65,
    "description_token": 0.22,
    "nested": 0.55,
    "fuzzy": 0.38,
    "event_type": 0.55,
    "upcoming_event": 0.35,
    "expired_event_penalty": -2.5,
    "concept_coverage": 0.85,
    "query_coverage": 2.4,
    "full_query_coverage": 1.1,
    "qualifier_coverage": 1.35,
    "multi_term_field_match": 0.9,
    "descriptive_concept": 0.75,
    "missing_event_type_penalty": -3.0,
    "distance": 0.5,
}

COVERAGE_FIELD_WEIGHTS = {
    "title": 1.0,
    "type": 0.95,
    "tags": 0.95,
    "label": 0.9,
    "subtitle": 0.85,
    "location": 0.8,
    "description": 0.78,
    "nested": 0.72,
    "host": 0.55,
}

_NORMALIZED_ALIAS_CACHE: dict[int, dict[str, list[tuple[str, set[str]]]]] = {}


def make_singular(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "zes")):
        return word[:-2]
    if word.endswith("ses") and not word.endswith("sses"):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = unescape(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"['`]", "", text)
    text = re.sub(r"[_/\\|:-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_list(text: Any, *, include_stop_words: bool = False) -> list[str]:
    tokens: list[str] = []
    for tok in normalize_text(text).split():
        singular = make_singular(tok)
        if len(singular) <= 1:
            continue
        if not include_stop_words and singular in STOP_WORDS:
            continue
        tokens.append(singular)
    return tokens


def tokenize(text: Any, *, include_stop_words: bool = False) -> set[str]:
    return set(tokenize_list(text, include_stop_words=include_stop_words))


def get_normalized_aliases(
    alias_map: dict[str, set[str]],
) -> dict[str, list[tuple[str, set[str]]]]:
    cache_key = id(alias_map)
    cached = _NORMALIZED_ALIAS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    normalized: dict[str, list[tuple[str, set[str]]]] = {}
    for canonical, aliases in alias_map.items():
        variants = []
        for alias in set(aliases) | {canonical}:
            alias_norm = normalize_text(alias)
            if not alias_norm:
                continue
            variants.append(
                (
                    alias_norm,
                    tokenize(alias_norm, include_stop_words=True),
                )
            )
        normalized[canonical] = variants

    _NORMALIZED_ALIAS_CACHE[cache_key] = normalized
    return normalized


def strip_urls(text: Any) -> str:
    if not text:
        return ""
    text = URL_RE.sub("", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def compact_description(text: Any, limit: int = MAX_DESC_CHARS) -> str:
    text = strip_urls(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def build_query_aware_description(
    encoded: dict[str, Any],
    query_hints: dict[str, Any],
    *,
    max_chars: int = MAX_DESCRIPTION_CONTEXT_CHARS,
) -> str:
    description = strip_urls(encoded["field_text"].get("description", ""))
    if not description:
        return ""
    if len(description) <= max_chars:
        return description
    return build_query_aware_text_excerpt(
        description,
        query_hints,
        max_chars=max_chars,
    )


def build_query_aware_text_excerpt(
    text: str,
    query_hints: dict[str, Any],
    *,
    max_chars: int,
) -> str:
    clean_text = strip_urls(text)
    if not clean_text:
        return ""
    if len(clean_text) <= max_chars:
        return clean_text

    spans = split_evidence_spans(clean_text)
    scored_spans = [
        (score_text_for_query_evidence(span, query_hints), index, span)
        for index, span in enumerate(spans)
        if span
    ]
    matching_spans = [
        item
        for item in scored_spans
        if item[0] > 0
    ]

    if not matching_spans:
        return clip_text_around_query(clean_text, query_hints, max_chars=max_chars)

    matching_spans.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = []
    selected_chars = 0
    for _, index, span in matching_spans:
        clipped = clip_text_around_query(span, query_hints, max_chars=max_chars)
        add_len = len(clipped) if not selected else len(clipped) + 5
        if selected and selected_chars + add_len > max_chars:
            continue
        if not selected and len(clipped) > max_chars:
            clipped = clip_text_around_query(clipped, query_hints, max_chars=max_chars)
            add_len = len(clipped)
        selected.append((index, clipped))
        selected_chars += add_len
        if len(selected) >= 4:
            break

    selected.sort(key=lambda item: item[0])
    return " ... ".join(span for _, span in selected)[:max_chars].rstrip()


def split_evidence_spans(text: str) -> list[str]:
    raw_parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    spans = []
    for part in raw_parts:
        part = re.sub(r"\s+", " ", part).strip()
        if not part:
            continue
        if len(part) <= 420:
            spans.append(part)
            continue
        line_parts = [
            re.sub(r"\s+", " ", chunk).strip()
            for chunk in re.split(r"\n+|;\s+", part)
            if chunk.strip()
        ]
        spans.extend(line_parts or [part])
    return spans


def score_text_for_query_evidence(
    text: str,
    query_hints: dict[str, Any],
) -> float:
    norm_text = normalize_text(text)
    tokens = tokenize(text)
    if not norm_text or not tokens:
        return 0.0

    score = 0.0
    expanded_hits = tokens & query_hints.get("expanded_tokens", set())
    score += len(expanded_hits) * 0.18

    for unit in query_hints.get("coverage_units", []):
        unit_tokens = set(unit.get("tokens", []))
        unit_phrases = unit.get("phrases", [])
        has_unit_match = bool(unit_tokens & tokens) or any(
            phrase in norm_text for phrase in unit_phrases
        )
        if has_unit_match:
            score += 1.0
            if unit.get("kind") == "qualifier":
                score += 0.45

    query_norm = query_hints.get("query_norm", "")
    if query_norm and query_norm in norm_text:
        score += 1.5

    return round(score, 6)


def clip_text_around_query(
    text: str,
    query_hints: dict[str, Any],
    *,
    max_chars: int,
) -> str:
    clean_text = re.sub(r"\s+", " ", text).strip()
    if len(clean_text) <= max_chars:
        return clean_text

    match_index = find_query_evidence_index(clean_text, query_hints)
    if match_index is None:
        return clean_text[:max_chars].rstrip() + "..."

    start = max(0, match_index - max_chars // 3)
    end = min(len(clean_text), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)

    if start > 0:
        boundary = clean_text.find(" ", start)
        if boundary != -1 and boundary < match_index:
            start = boundary + 1
    if end < len(clean_text):
        boundary = clean_text.rfind(" ", start, end)
        if boundary > start:
            end = boundary

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean_text) else ""
    return f"{prefix}{clean_text[start:end].strip()}{suffix}"


def find_query_evidence_index(
    text: str,
    query_hints: dict[str, Any],
) -> int | None:
    lower_text = text.lower()
    candidates: list[int] = []

    for unit in query_hints.get("coverage_units", []):
        for phrase in unit.get("phrases", []):
            index = lower_text.find(phrase)
            if index >= 0:
                candidates.append(index)
        for token in unit.get("tokens", []):
            match = re.search(rf"\b{re.escape(token)}\b", lower_text)
            if match:
                candidates.append(match.start())

    for token in query_hints.get("expanded_tokens", set()):
        match = re.search(rf"\b{re.escape(token)}\b", lower_text)
        if match:
            candidates.append(match.start())

    return min(candidates) if candidates else None


def compact_nested_value(value: Any) -> str:
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


def join_with_limit(parts: list[str], max_chars: int) -> str:
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


def detect_canonical_matches(text: Any, alias_map: dict[str, set[str]]) -> set[str]:
    norm_text = normalize_text(text)
    raw_tokens = tokenize(norm_text, include_stop_words=True)
    found = set()

    for canonical, variants in get_normalized_aliases(alias_map).items():
        for alias_norm, alias_tokens in variants:
            if " " in alias_norm:
                if alias_norm in norm_text:
                    found.add(canonical)
                    break
            elif alias_norm in raw_tokens or make_singular(alias_norm) in raw_tokens:
                found.add(canonical)
                break
            elif alias_tokens and alias_tokens.issubset(raw_tokens):
                found.add(canonical)
                break
    return found


def build_alias_coverage_unit(
    kind: str,
    name: str,
    alias_map: dict[str, set[str]],
    expansion_map: dict[str, set[str]] | None = None,
    *,
    include_phrase_tokens: bool = True,
) -> dict[str, Any]:
    aliases = set(alias_map.get(name, set())) | {name}
    if expansion_map:
        aliases.update(expansion_map.get(name, set()))

    phrases = set()
    tokens = set()
    for alias in aliases:
        alias_norm = normalize_text(alias)
        if not alias_norm:
            continue
        if " " in alias_norm:
            phrases.add(alias_norm)
            if include_phrase_tokens:
                tokens.update(tokenize_list(alias_norm))
        else:
            tokens.update(tokenize_list(alias_norm))

    return {
        "kind": kind,
        "name": name,
        "tokens": sorted(tokens),
        "phrases": sorted(phrases),
    }


def build_query_coverage_units(
    scoring_tokens: set[str],
    concepts: set[str],
    qualifiers: set[str],
) -> list[dict[str, Any]]:
    units = []
    consumed_tokens = set()

    for concept in sorted(concepts):
        unit = build_alias_coverage_unit(
            "concept",
            concept,
            CONCEPT_ALIASES,
            CONCEPT_QUERY_EXPANSIONS,
        )
        consumed_tokens.update(scoring_tokens & set(unit["tokens"]))
        units.append(unit)

    for qualifier in sorted(qualifiers):
        unit = build_alias_coverage_unit(
            "qualifier",
            qualifier,
            QUALIFIER_MATCH_ALIASES,
            QUALIFIER_QUERY_EXPANSIONS,
            include_phrase_tokens=False,
        )
        detect_unit = build_alias_coverage_unit(
            "qualifier",
            qualifier,
            QUALIFIER_ALIASES,
            QUALIFIER_QUERY_EXPANSIONS,
        )
        consumed_tokens.update(scoring_tokens & set(detect_unit["tokens"]))
        units.append(unit)

    for token in sorted(scoring_tokens - consumed_tokens):
        token_variants = {token}
        token_variants.update(TOKEN_WORD_BANK.get(token, set()))
        token_variants = {
            make_singular(variant)
            for variant in token_variants
            if make_singular(variant) not in STOP_WORDS
        }
        if token_variants:
            units.append({
                "kind": "token",
                "name": token,
                "tokens": sorted(token_variants),
                "phrases": [],
            })

    return units


def build_query_hints(query: str) -> dict[str, Any]:
    query_norm = normalize_text(query)
    raw_tokens = tokenize(query_norm, include_stop_words=True)
    scoring_tokens = tokenize(query_norm)
    expanded_tokens = set(scoring_tokens)

    for phrase, expansions in PHRASE_WORD_BANK.items():
        if normalize_text(phrase) in query_norm:
            expanded_tokens.update(make_singular(token) for token in expansions)

    for tok in list(expanded_tokens):
        expanded_tokens.update(TOKEN_WORD_BANK.get(tok, set()))

    days = detect_canonical_matches(query, DAY_ALIASES)
    meals = detect_canonical_matches(query, MEAL_ALIASES)
    locations = detect_canonical_matches(query, LOCATION_ALIASES)
    concepts = detect_canonical_matches(query, CONCEPT_ALIASES)
    qualifiers = detect_canonical_matches(query, QUALIFIER_ALIASES)

    for canonical in days:
        expanded_tokens.add(canonical)
        expanded_tokens.update(DAY_ALIASES.get(canonical, set()))

    for canonical in meals:
        expanded_tokens.add(canonical)
        expanded_tokens.update(MEAL_ALIASES.get(canonical, set()))

    for canonical in locations:
        expanded_tokens.add(canonical)
        expanded_tokens.update(LOCATION_ALIASES.get(canonical, set()))

    for canonical in concepts:
        expanded_tokens.add(canonical)
        expanded_tokens.update(
            make_singular(token)
            for alias in CONCEPT_QUERY_EXPANSIONS.get(canonical, set())
            for token in tokenize_list(alias, include_stop_words=True)
        )

    for canonical in qualifiers:
        expanded_tokens.add(canonical)
        expanded_tokens.update(
            make_singular(token)
            for alias in QUALIFIER_QUERY_EXPANSIONS.get(canonical, set())
            for token in tokenize_list(alias, include_stop_words=True)
        )

    if "free" in qualifiers:
        expanded_tokens.difference_update({"cost", "money", "charge", "doesnt", "does", "not"})

    expanded_tokens = {
        make_singular(token)
        for token in expanded_tokens
        if len(make_singular(token)) > 1 and make_singular(token) not in STOP_WORDS
    }
    coverage_units = build_query_coverage_units(scoring_tokens, concepts, qualifiers)

    return {
        "query": query,
        "query_norm": query_norm,
        "raw_tokens": raw_tokens,
        "scoring_tokens": scoring_tokens,
        "expanded_tokens": expanded_tokens,
        "days": days,
        "meals": meals,
        "locations": locations,
        "concepts": concepts,
        "qualifiers": qualifiers,
        "coverage_units": coverage_units,
        "has_proximity_intent": any(
            normalize_text(alias) in query_norm for alias in PROXIMITY_ALIASES
        ),
    }


def canonicalize_from_aliases(value: Any, alias_map: dict[str, set[str]]) -> str:
    matches = detect_canonical_matches(value or "", alias_map)
    if matches:
        return sorted(matches)[0]
    return normalize_text(value or "")


def gather_generic_strings(obj: Any, fragments: list[str]) -> None:
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


def make_segment_text(
    day_title: str,
    tab_title: str,
    header: str,
    extra_parts: list[Any],
) -> str:
    parts = []
    if day_title:
        parts.append(f"day={day_title}")
    if tab_title:
        parts.append(f"tab={tab_title}")
    if header:
        parts.append(f"item={header}")

    for part in extra_parts:
        compact = compact_nested_value(part)
        if compact:
            parts.append(compact)

    deduped = []
    seen = set()
    for part in parts:
        norm = normalize_text(part)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(part)

    return " || ".join(deduped).strip()


def extract_nested_segments(nested_content: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    seq = 0

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
                                if (
                                    key in {"header", "bullets"}
                                    or key in SKIP_NESTED_KEYS
                                ):
                                    continue
                                if isinstance(value, str):
                                    extra_parts.append(f"{key}={value}")

                            text = make_segment_text(
                                day_title, tab_title, header, extra_parts
                            )
                            if not text:
                                continue

                            segments.append(_make_segment(seq, text, day_title, tab_title, header))
                            seq += 1
                    else:
                        fallback_parts: list[str] = []
                        gather_generic_strings(tab_obj, fallback_parts)
                        text = make_segment_text(day_title, tab_title, "", fallback_parts)
                        if text:
                            segments.append(_make_segment(seq, text, day_title, tab_title, ""))
                            seq += 1
            else:
                fallback_parts = []
                gather_generic_strings(day_obj, fallback_parts)
                text = make_segment_text(day_title, "", "", fallback_parts)
                if text:
                    segments.append(_make_segment(seq, text, day_title, "", ""))
                    seq += 1
    else:
        fallback_parts = []
        gather_generic_strings(nested_content, fallback_parts)
        text = make_segment_text("", "", "", fallback_parts)
        if text:
            segments.append(_make_segment(seq, text, "", "", ""))

    return segments


def _make_segment(
    idx: int,
    text: str,
    day_title: str,
    tab_title: str,
    header: str,
) -> dict[str, Any]:
    return {
        "idx": idx,
        "day": day_title,
        "tab": tab_title,
        "header": header,
        "text": text,
        "normalized": normalize_text(text),
        "token_set": tokenize(text),
        "token_list": tokenize_list(text),
        "day_canonical": canonicalize_from_aliases(day_title, DAY_ALIASES),
        "tab_canonical": canonicalize_from_aliases(tab_title, MEAL_ALIASES),
        "concepts": detect_canonical_matches(text, CONCEPT_ALIASES),
    }


def collapse_structured_segments(
    segments: list[dict[str, Any]],
    max_chars: int = MAX_NESTED_SEARCH_CHARS,
) -> str:
    return join_with_limit([seg["text"] for seg in segments], max_chars)


def score_segment(query_hints: dict[str, Any], segment: dict[str, Any]) -> float:
    expanded_tokens = query_hints["expanded_tokens"]
    query_norm = query_hints["query_norm"]

    if not query_norm:
        return 0.0

    seg_norm = segment["normalized"][:MAX_SEGMENT_FOR_FUZZY]
    seg_tokens = segment["token_set"]

    overlap = len(expanded_tokens & seg_tokens)
    overlap_ratio = overlap / max(len(expanded_tokens), 1)

    contains_boost = 1.0 if query_norm in seg_norm else 0.0

    day_bonus = 0.0
    meal_bonus = 0.0
    concept_bonus = 0.0

    if query_hints["days"] and segment["day_canonical"] in query_hints["days"]:
        day_bonus += 0.60
    if query_hints["meals"] and segment["tab_canonical"] in query_hints["meals"]:
        meal_bonus += 0.50
    if query_hints["concepts"] & segment["concepts"]:
        concept_bonus += 0.35

    header_tokens = tokenize(segment.get("header", ""))
    header_hits = len(expanded_tokens & header_tokens)
    has_direct_evidence = (
        overlap
        or contains_boost
        or day_bonus
        or meal_bonus
        or concept_bonus
        or header_hits
    )

    if not has_direct_evidence:
        return 0.0

    seq_ratio = SequenceMatcher(None, query_norm, seg_norm).ratio()

    score = (
        (overlap_ratio * 0.35)
        + (seq_ratio * 0.12)
        + (contains_boost * 0.15)
        + day_bonus
        + meal_bonus
        + concept_bonus
        + (header_hits * 0.08)
    )

    return round(score, 6)


def build_query_aware_nested_excerpt(
    segments: list[dict[str, Any]],
    query_hints: dict[str, Any],
    max_chars: int = MAX_NESTED_CONTEXT_CHARS,
) -> str:
    if not segments:
        return ""

    scored = []
    for seg in segments:
        seg_score = score_segment(query_hints, seg)
        scored.append((seg_score, seg["idx"], seg))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    selected = []
    selected_chars = 0
    group_counts: dict[tuple[str, str], int] = {}

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


def encode_item(item: dict[str, Any]) -> dict[str, Any]:
    title = _read_string(item.get("title"))
    subtitle = _read_string(item.get("subtitle"))
    host = _read_string(item.get("host"))
    description = _read_string(item.get("description"))
    tags = _read_string_list(item.get("tags"))
    item_type = _read_string(item.get("type"))
    location = _read_string(item.get("location"))
    location_at = _read_string(item.get("location_at"))
    label_name = ""

    label = item.get("label")
    if isinstance(label, dict):
        label_name = _read_string(label.get("name"))

    nested_sources: list[Any] = []
    for key in ("nested_content", "sections"):
        nested = item.get(key)
        if nested:
            nested_sources.append(nested)

    nested_segments: list[dict[str, Any]] = []
    for nested in nested_sources:
        nested_segments.extend(extract_nested_segments(nested))

    nested_structured_text = collapse_structured_segments(nested_segments)
    tags_text = " ".join(tags)
    item_type_text = item_type.replace("_", " ")
    location_text = " ".join(part for part in [location, location_at, label_name] if part)

    field_text = {
        "title": title,
        "subtitle": subtitle,
        "host": host,
        "description": description,
        "tags": tags_text,
        "type": item_type_text,
        "location": location_text,
        "label": label_name,
        "nested": nested_structured_text,
    }
    field_tokens = {
        field: tokenize_list(text)
        for field, text in field_text.items()
    }
    field_token_sets = {
        field: set(tokens)
        for field, tokens in field_tokens.items()
    }

    search_blob = " | ".join(
        part
        for part in [
            title,
            subtitle,
            host,
            tags_text,
            description,
            nested_structured_text,
            item_type_text,
            location_text,
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
        "location": location,
        "location_at": location_at,
        "location_id": item.get("location_id"),
        "start": item.get("start", "") or "",
        "end": item.get("end", "") or "",
        "source_url": item.get("source_url", "") or "",
    }

    return {
        "raw": item,
        "compact": compact_item,
        "search_blob": search_blob,
        "normalized_blob": normalize_text(search_blob),
        "token_set": tokenize(search_blob),
        "token_list": tokenize_list(search_blob),
        "field_text": field_text,
        "field_tokens": field_tokens,
        "field_token_sets": field_token_sets,
        "nested_segments": nested_segments,
        "nested_structured_text": nested_structured_text,
        "title_concepts": detect_canonical_matches(title, CONCEPT_ALIASES),
        "type_concepts": detect_canonical_matches(item_type_text, CONCEPT_ALIASES),
        "tag_concepts": detect_canonical_matches(tags_text, CONCEPT_ALIASES),
        "subtitle_concepts": detect_canonical_matches(subtitle, CONCEPT_ALIASES),
        "description_concepts": detect_canonical_matches(description, CONCEPT_ALIASES),
        "location_concepts": detect_canonical_matches(location_text, CONCEPT_ALIASES),
        "geometry": item.get("geometry"),
        "is_event": normalize_text(item_type) == "event",
    }


def rank_items(
    query: str,
    items: list[dict[str, Any]],
    *,
    user_location: Any = None,
    max_context_items: int = MAX_CONTEXT_ITEMS,
    min_context_items: int = MIN_CONTEXT_ITEMS,
    max_context_tokens: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    query_hints = build_query_hints(query)
    encoded_items = [encode_item(item) for item in items if item.get("id")]
    bm25_stats = build_bm25_stats(encoded_items)
    parsed_user_location = parse_user_location(user_location)
    now = now or datetime.now(timezone.utc)

    ranked_rows = []
    for encoded in encoded_items:
        row = score_encoded_item(
            query_hints,
            encoded,
            bm25_stats=bm25_stats,
            user_location=parsed_user_location,
            now=now,
        )
        ranked_rows.append(row)

    ranked_rows.sort(
        key=lambda row: (
            row["score"],
            row["signals"].get("structured_concept", 0),
            row["signals"].get("exact_title", 0),
            row["signals"].get("bm25", 0),
        ),
        reverse=True,
    )

    for index, row in enumerate(ranked_rows, start=1):
        row["rank"] = index

    context_rows = select_context_rows(ranked_rows, max_context_items, min_context_items)
    llm_candidates = build_llm_candidates(context_rows, query_hints)
    if max_context_tokens is not None:
        context_rows, llm_candidates = trim_context_to_token_budget(
            context_rows,
            query_hints,
            max_context_tokens=max_context_tokens,
            min_context_items=min_context_items,
        )

    return {
        "query_hints": query_hints,
        "ranked_rows": ranked_rows,
        "context_rows": context_rows,
        "llm_candidates": llm_candidates,
        "context_token_estimate": approximate_json_tokens(llm_candidates),
    }


def build_bm25_stats(encoded_items: list[dict[str, Any]]) -> dict[str, Any]:
    field_stats: dict[str, dict[str, Any]] = {}
    total_docs = len(encoded_items)

    for field in FIELD_WEIGHTS:
        lengths = []
        doc_freqs: Counter[str] = Counter()
        for encoded in encoded_items:
            tokens = encoded["field_tokens"].get(field, [])
            lengths.append(len(tokens))
            doc_freqs.update(set(tokens))

        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        field_stats[field] = {
            "avg_len": avg_len,
            "doc_freqs": doc_freqs,
        }

    return {
        "total_docs": total_docs,
        "fields": field_stats,
    }


def score_encoded_item(
    query_hints: dict[str, Any],
    encoded: dict[str, Any],
    *,
    bm25_stats: dict[str, Any] | None = None,
    user_location: tuple[float, float] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    query_norm = query_hints["query_norm"]
    expanded_tokens = query_hints["expanded_tokens"]

    if not query_norm:
        score = 0.0
        return _make_scored_row(encoded, score, {}, False, set(), [], [])

    matched_concepts = query_hints["concepts"] & get_item_concepts(encoded)
    strong_concept_fields = (
        encoded["title_concepts"]
        | encoded["type_concepts"]
        | encoded["tag_concepts"]
        | encoded["location_concepts"]
    )
    strong_matched_concepts = query_hints["concepts"] & strong_concept_fields

    signals: dict[str, float] = {}
    field_sets = encoded["field_token_sets"]
    title_tokens = field_sets.get("title", set())
    type_tokens = field_sets.get("type", set())
    tag_tokens = field_sets.get("tags", set())
    location_tokens = field_sets.get("location", set())
    desc_tokens = field_sets.get("description", set())

    if bm25_stats:
        signals["bm25"] = calculate_weighted_bm25(expanded_tokens, encoded, bm25_stats)

    if query_norm and query_norm == encoded["field_text"].get("title", "").lower().strip():
        signals["exact_title"] = 1.0

    if title_tokens and title_tokens.issubset(expanded_tokens):
        signals["all_title_terms"] = 1.0

    title_token_ratio = len(expanded_tokens & title_tokens) / max(len(title_tokens), 1)
    if title_token_ratio:
        signals["title_token"] = title_token_ratio

    if strong_matched_concepts:
        signals["structured_concept"] = float(len(strong_matched_concepts))
    if query_hints["concepts"] & encoded["title_concepts"]:
        signals["title_concept"] = float(len(query_hints["concepts"] & encoded["title_concepts"]))
    if query_hints["concepts"] & encoded["tag_concepts"]:
        signals["tag_concept"] = float(len(query_hints["concepts"] & encoded["tag_concepts"]))

    if query_hints["concepts"]:
        signals["concept_coverage"] = len(matched_concepts) / len(query_hints["concepts"])

    nested_concepts = set().union(*(seg["concepts"] for seg in encoded["nested_segments"]))
    descriptive_only_concepts = (
        query_hints["concepts"]
        & (encoded["description_concepts"] | nested_concepts)
    ) - strong_concept_fields
    if descriptive_only_concepts:
        signals["descriptive_concept"] = float(len(descriptive_only_concepts))

    coverage_signals, matched_query_units, unmatched_query_units = (
        calculate_query_coverage_signals(query_hints, encoded)
    )
    signals.update(coverage_signals)

    location_ratio = len(expanded_tokens & location_tokens) / max(len(expanded_tokens), 1)
    if location_ratio:
        signals["location_token"] = location_ratio

    description_ratio = len(expanded_tokens & desc_tokens) / max(len(expanded_tokens), 1)
    if description_ratio:
        signals["description_token"] = description_ratio

    nested_scores = [
        score_segment(query_hints, seg)
        for seg in encoded["nested_segments"]
    ]
    if nested_scores:
        signals["nested"] = max(nested_scores)
        nested_match_count = sum(1 for score in nested_scores if score >= 0.40)
        if nested_match_count:
            signals["nested_match_count"] = min(nested_match_count, 3) * 0.07

    fuzzy_score = calculate_supporting_fuzzy_score(query_hints, encoded)
    if fuzzy_score:
        signals["fuzzy"] = fuzzy_score

    if "event" in query_hints["concepts"] and encoded["is_event"]:
        signals["event_type"] = 1.0
        event_state = get_event_state(encoded["compact"], now or datetime.now(timezone.utc))
        if event_state == "upcoming":
            signals["upcoming_event"] = 1.0
        elif event_state == "expired":
            signals["expired_event_penalty"] = 1.0
    elif "event" in query_hints["concepts"] and len(query_hints["concepts"]) > 1:
        signals["missing_event_type_penalty"] = 1.0

    if query_hints["has_proximity_intent"] and user_location:
        distance_score = calculate_distance_score(user_location, encoded.get("geometry"))
        if distance_score is not None:
            signals["distance"] = distance_score

    # Exact type/tag tokens are structured evidence even when a concept alias was
    # not detected, e.g. an administrator adds a new type name already queried.
    structured_token_ratio = len(expanded_tokens & (type_tokens | tag_tokens)) / max(
        len(expanded_tokens),
        1,
    )
    if structured_token_ratio:
        signals["structured_token"] = structured_token_ratio

    score = (
        signals.get("bm25", 0.0) * SIGNAL_WEIGHTS["bm25"]
        + signals.get("exact_title", 0.0) * SIGNAL_WEIGHTS["exact_title"]
        + signals.get("all_title_terms", 0.0) * SIGNAL_WEIGHTS["all_title_terms"]
        + signals.get("title_token", 0.0) * SIGNAL_WEIGHTS["title_token"]
        + signals.get("structured_concept", 0.0) * SIGNAL_WEIGHTS["structured_concept"]
        + signals.get("title_concept", 0.0) * SIGNAL_WEIGHTS["title_concept"]
        + signals.get("tag_concept", 0.0) * SIGNAL_WEIGHTS["tag_concept"]
        + signals.get("location_token", 0.0) * SIGNAL_WEIGHTS["location_token"]
        + signals.get("description_token", 0.0) * SIGNAL_WEIGHTS["description_token"]
        + signals.get("nested", 0.0) * SIGNAL_WEIGHTS["nested"]
        + signals.get("nested_match_count", 0.0)
        + signals.get("fuzzy", 0.0) * SIGNAL_WEIGHTS["fuzzy"]
        + signals.get("event_type", 0.0) * SIGNAL_WEIGHTS["event_type"]
        + signals.get("upcoming_event", 0.0) * SIGNAL_WEIGHTS["upcoming_event"]
        + signals.get("expired_event_penalty", 0.0) * SIGNAL_WEIGHTS["expired_event_penalty"]
        + signals.get("concept_coverage", 0.0) * SIGNAL_WEIGHTS["concept_coverage"]
        + signals.get("query_coverage", 0.0) * SIGNAL_WEIGHTS["query_coverage"]
        + signals.get("full_query_coverage", 0.0) * SIGNAL_WEIGHTS["full_query_coverage"]
        + signals.get("qualifier_coverage", 0.0) * SIGNAL_WEIGHTS["qualifier_coverage"]
        + signals.get("multi_term_field_match", 0.0) * SIGNAL_WEIGHTS["multi_term_field_match"]
        + signals.get("descriptive_concept", 0.0) * SIGNAL_WEIGHTS["descriptive_concept"]
        + signals.get("missing_event_type_penalty", 0.0) * SIGNAL_WEIGHTS["missing_event_type_penalty"]
        + signals.get("distance", 0.0) * SIGNAL_WEIGHTS["distance"]
        + signals.get("structured_token", 0.0) * 0.85
    )

    protected_recall = bool(strong_matched_concepts)
    return _make_scored_row(
        encoded,
        round(score, 6),
        signals,
        protected_recall,
        matched_concepts,
        matched_query_units,
        unmatched_query_units,
    )


def calculate_weighted_bm25(
    query_tokens: set[str],
    encoded: dict[str, Any],
    bm25_stats: dict[str, Any],
) -> float:
    total_docs = bm25_stats.get("total_docs", 0)
    if total_docs <= 0 or not query_tokens:
        return 0.0

    raw_score = 0.0
    for field, field_weight in FIELD_WEIGHTS.items():
        tokens = encoded["field_tokens"].get(field, [])
        field_score = bm25(query_tokens, tokens, bm25_stats["fields"][field], total_docs)
        raw_score += field_score * field_weight

    return round(math.log1p(raw_score), 6)


def bm25(
    query_tokens: set[str],
    doc_tokens: list[str],
    field_stats: dict[str, Any],
    total_docs: int,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    if not doc_tokens:
        return 0.0

    token_counts = Counter(doc_tokens)
    avg_len = field_stats.get("avg_len") or 1.0
    doc_len = len(doc_tokens)
    doc_freqs = field_stats.get("doc_freqs", Counter())
    score = 0.0

    for token in query_tokens:
        tf = token_counts.get(token, 0)
        if tf <= 0:
            continue

        df = doc_freqs.get(token, 0)
        idf = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
        denom = tf + k1 * (1 - b + b * (doc_len / avg_len))
        score += idf * ((tf * (k1 + 1)) / denom)

    return score


def calculate_supporting_fuzzy_score(
    query_hints: dict[str, Any],
    encoded: dict[str, Any],
) -> float:
    query_norm = query_hints["query_norm"]
    query_tokens = query_hints["scoring_tokens"]
    if not query_norm:
        return 0.0

    best_field_ratio = 0.0
    for field in ("title", "subtitle", "location", "tags", "type", "host"):
        field_text = normalize_text(encoded["field_text"].get(field, ""))
        if not field_text:
            continue
        ratio = SequenceMatcher(None, query_norm, field_text).ratio()
        if ratio > best_field_ratio:
            best_field_ratio = ratio

    token_ratio = 0.0
    candidate_tokens = (
        encoded["field_token_sets"].get("title", set())
        | encoded["field_token_sets"].get("subtitle", set())
        | encoded["field_token_sets"].get("location", set())
        | encoded["field_token_sets"].get("tags", set())
        | encoded["field_token_sets"].get("type", set())
    )
    for query_token in query_tokens:
        for candidate_token in candidate_tokens:
            if abs(len(query_token) - len(candidate_token)) > 2:
                continue
            ratio = SequenceMatcher(None, query_token, candidate_token).ratio()
            if ratio > token_ratio:
                token_ratio = ratio

    field_signal = max(0.0, best_field_ratio - 0.60) / 0.40
    token_signal = max(0.0, token_ratio - 0.82) / 0.18
    return round(min(1.0, max(field_signal, token_signal)), 6)


def calculate_query_coverage_signals(
    query_hints: dict[str, Any],
    encoded: dict[str, Any],
) -> tuple[dict[str, float], list[str], list[str]]:
    units = query_hints.get("coverage_units", [])
    if not units:
        return {}, [], []

    unit_scores = []
    matched_units = []
    unmatched_units = []
    qualifier_scores = []
    field_unit_counts: Counter[str] = Counter()

    for unit in units:
        evidence_score, fields = calculate_unit_evidence(unit, encoded)
        unit_name = f"{unit['kind']}:{unit['name']}"
        if evidence_score > 0:
            unit_scores.append(evidence_score)
            matched_units.append(unit_name)
            for field in fields:
                field_unit_counts[field] += 1
        else:
            unit_scores.append(0.0)
            unmatched_units.append(unit_name)

        if unit["kind"] == "qualifier":
            qualifier_scores.append(evidence_score)

    signals = {
        "query_coverage": round(sum(unit_scores) / len(units), 6),
    }

    if all(score > 0 for score in unit_scores):
        signals["full_query_coverage"] = 1.0

    if qualifier_scores:
        signals["qualifier_coverage"] = round(
            sum(qualifier_scores) / len(qualifier_scores),
            6,
        )

    if len(units) > 1 and field_unit_counts:
        best_field_score = max(
            (count / len(units)) * COVERAGE_FIELD_WEIGHTS.get(field, 0.5)
            for field, count in field_unit_counts.items()
        )
        if best_field_score > 0:
            signals["multi_term_field_match"] = round(best_field_score, 6)

    return signals, matched_units, unmatched_units


def calculate_unit_evidence(
    unit: dict[str, Any],
    encoded: dict[str, Any],
) -> tuple[float, list[str]]:
    best_score = 0.0
    matched_fields = []
    unit_tokens = set(unit.get("tokens", []))
    unit_phrases = unit.get("phrases", [])

    for field, field_weight in COVERAGE_FIELD_WEIGHTS.items():
        field_text = encoded["field_text"].get(field, "")
        field_norm = normalize_text(field_text)
        field_tokens = encoded["field_token_sets"].get(field, set())

        has_token_match = bool(unit_tokens & field_tokens)
        has_phrase_match = any(phrase in field_norm for phrase in unit_phrases)
        has_concept_match = unit_matches_field_concept(unit, encoded, field)

        if has_token_match or has_phrase_match or has_concept_match:
            best_score = max(best_score, field_weight)
            matched_fields.append(field)

    return round(best_score, 6), matched_fields


def unit_matches_field_concept(
    unit: dict[str, Any],
    encoded: dict[str, Any],
    field: str,
) -> bool:
    if unit.get("kind") != "concept":
        return False

    concept = unit.get("name")
    if field == "title":
        return concept in encoded["title_concepts"]
    if field == "type":
        return concept in encoded["type_concepts"]
    if field == "tags":
        return concept in encoded["tag_concepts"]
    if field == "subtitle":
        return concept in encoded["subtitle_concepts"]
    if field == "description":
        return concept in encoded["description_concepts"]
    if field in {"location", "label"}:
        return concept in encoded["location_concepts"]
    if field == "nested":
        return any(concept in segment["concepts"] for segment in encoded["nested_segments"])
    return False


def select_context_rows(
    ranked_rows: list[dict[str, Any]],
    max_context_items: int,
    min_context_items: int,
) -> list[dict[str, Any]]:
    if not ranked_rows:
        return []

    selected = []
    selected_ids = {row["id"] for row in selected}
    diversity_counts: Counter[str] = Counter()
    duplicate_skips = []

    for row in ranked_rows:
        if len(selected) >= max_context_items:
            break
        if row["score"] <= 0 and len(selected) >= min_context_items:
            break

        diversity_key = get_context_diversity_key(row)
        if (
            diversity_key
            and diversity_counts[diversity_key] >= MAX_CONTEXT_DUPLICATES_PER_KEY
        ):
            duplicate_skips.append(row)
            continue

        selected.append(row)
        selected_ids.add(row["id"])
        if diversity_key:
            diversity_counts[diversity_key] += 1

    protected_rows = [
        row
        for row in ranked_rows
        if row["protected_recall"] and row["id"] not in selected_ids
    ]

    for protected_row in protected_rows:
        if len(selected) < max_context_items:
            selected.append(protected_row)
            selected_ids.add(protected_row["id"])
            continue

        replace_index = None
        for idx in range(len(selected) - 1, -1, -1):
            if not selected[idx]["protected_recall"]:
                replace_index = idx
                break

        if replace_index is None:
            break

        removed_row = selected[replace_index]
        selected_ids.discard(removed_row["id"])
        selected[replace_index] = protected_row
        selected_ids.add(protected_row["id"])

    if len(selected) < min_context_items:
        for row in [*duplicate_skips, *ranked_rows]:
            if row["id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["id"])
            if len(selected) >= min_context_items:
                break

    selected.sort(key=lambda row: row["rank"])
    return selected[:max_context_items]


def get_context_diversity_key(row: dict[str, Any]) -> str:
    compact = row.get("compact", {})
    title = normalize_diversity_text(compact.get("title", ""))
    item_type = normalize_text(compact.get("type", ""))
    host = normalize_diversity_text(compact.get("host", ""))
    if not title:
        return ""
    return "|".join(part for part in (item_type, host, title) if part)


def normalize_diversity_text(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)
    text = re.sub(r"\b(day|session|part|week)\s+\d+\b", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_llm_candidates(
    context_rows: list[dict[str, Any]],
    query_hints: dict[str, Any],
    *,
    description_max_chars: int = MAX_DESCRIPTION_CONTEXT_CHARS,
    nested_max_chars: int = MAX_NESTED_CONTEXT_CHARS,
) -> list[dict[str, Any]]:
    llm_candidates = []
    for row in context_rows:
        encoded = row["encoded"]
        compact = dict(encoded["compact"])
        description = build_query_aware_description(
            encoded,
            query_hints,
            max_chars=description_max_chars,
        )
        if description:
            compact["description"] = description
        compact["local_rank"] = row["rank"]
        compact["retrieval_score"] = row["score"]
        if row["matched_concepts"]:
            compact["matched_concepts"] = sorted(row["matched_concepts"])
        if row["matched_query_units"]:
            compact["matched_query_units"] = row["matched_query_units"]
        if row["unmatched_query_units"]:
            compact["unmatched_query_units"] = row["unmatched_query_units"]

        nested_compact = build_query_aware_nested_excerpt(
            encoded["nested_segments"],
            query_hints,
            max_chars=nested_max_chars,
        )

        if nested_compact:
            compact["nested_content_compact"] = nested_compact

        evidence_parts = []
        if description:
            evidence_parts.append(
                build_query_aware_text_excerpt(
                    description,
                    query_hints,
                    max_chars=MAX_DESCRIPTION_EXCERPT_CHARS,
                )
            )
        if nested_compact:
            evidence_parts.append(
                build_query_aware_text_excerpt(
                    nested_compact,
                    query_hints,
                    max_chars=MAX_NESTED_EVIDENCE_CHARS,
                )
            )
        evidence_excerpt = join_with_limit(
            [part for part in evidence_parts if part],
            MAX_EVIDENCE_EXCERPT_CHARS,
        )
        if evidence_excerpt:
            compact["evidence_excerpt"] = evidence_excerpt

        llm_candidates.append(compact)
    return llm_candidates


def trim_context_to_token_budget(
    context_rows: list[dict[str, Any]],
    query_hints: dict[str, Any],
    *,
    max_context_tokens: int,
    min_context_items: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trimmed_rows = list(context_rows)
    llm_candidates = build_llm_candidates(trimmed_rows, query_hints)

    if approximate_json_tokens(llm_candidates) > max_context_tokens:
        llm_candidates = build_llm_candidates(
            trimmed_rows,
            query_hints,
            description_max_chars=MAX_DESCRIPTION_EXCERPT_CHARS,
            nested_max_chars=MAX_NESTED_EVIDENCE_CHARS,
        )

    while (
        approximate_json_tokens(llm_candidates) > max_context_tokens
        and len(trimmed_rows) > min_context_items
    ):
        remove_index = select_context_trim_remove_index(trimmed_rows)
        trimmed_rows.pop(remove_index)
        llm_candidates = build_llm_candidates(
            trimmed_rows,
            query_hints,
            description_max_chars=MAX_DESCRIPTION_EXCERPT_CHARS,
            nested_max_chars=MAX_NESTED_EVIDENCE_CHARS,
        )

    return trimmed_rows, llm_candidates


def select_context_trim_remove_index(context_rows: list[dict[str, Any]]) -> int:
    return min(
        range(len(context_rows)),
        key=lambda index: (
            context_trim_keep_score(context_rows[index]),
            -(context_rows[index].get("rank") or 0),
        ),
    )


def context_trim_keep_score(row: dict[str, Any]) -> float:
    signals = row.get("signals", {})
    return (
        row.get("score", 0.0)
        + signals.get("query_coverage", 0.0) * 5.0
        + signals.get("full_query_coverage", 0.0) * 2.0
        + signals.get("qualifier_coverage", 0.0) * 1.5
        + signals.get("multi_term_field_match", 0.0) * 1.0
        + signals.get("description_token", 0.0) * 1.0
        + signals.get("nested", 0.0) * 1.0
        + len(row.get("matched_query_units", [])) * 0.45
        - len(row.get("unmatched_query_units", [])) * 0.20
        + (1.0 if row.get("protected_recall") else 0.0)
    )


def get_item_concepts(encoded: dict[str, Any]) -> set[str]:
    return (
        encoded["title_concepts"]
        | encoded["type_concepts"]
        | encoded["tag_concepts"]
        | encoded["subtitle_concepts"]
        | encoded["description_concepts"]
        | encoded["location_concepts"]
        | set().union(*(seg["concepts"] for seg in encoded["nested_segments"]))
    )


def get_event_state(compact: dict[str, Any], now: datetime) -> str:
    if compact.get("type") != "event":
        return "not_event"

    end = parse_datetime(compact.get("end"))
    start = parse_datetime(compact.get("start"))

    if end and end <= now:
        return "expired"
    if start and start >= now:
        return "upcoming"
    if start and (not end or end > now):
        return "current"
    return "unknown"


def parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_user_location(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    latitude = value.get("latitude", value.get("lat"))
    longitude = value.get("longitude", value.get("lng", value.get("lon")))
    if isinstance(latitude, bool) or isinstance(longitude, bool):
        return None
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return None
    lat = float(latitude)
    lon = float(longitude)
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        return None
    return lat, lon


def calculate_distance_score(
    user_location: tuple[float, float],
    geometry: Any,
) -> float | None:
    item_location = extract_item_lat_lon(geometry)
    if not item_location:
        return None

    distance_meters = haversine_meters(user_location, item_location)
    if distance_meters <= 0:
        return 1.0
    return round(1 / (1 + (distance_meters / 350)), 6)


def extract_item_lat_lon(geometry: Any) -> tuple[float, float] | None:
    if not isinstance(geometry, dict):
        return None

    latitude = geometry.get("latitude")
    longitude = geometry.get("longitude")

    if latitude is None and longitude is None:
        coordinates = geometry.get("coordinates")
        if (
            isinstance(coordinates, list)
            and len(coordinates) >= 2
            and all(isinstance(value, (int, float)) for value in coordinates[:2])
        ):
            longitude, latitude = coordinates[:2]

    if isinstance(latitude, bool) or isinstance(longitude, bool):
        return None
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return None

    return float(latitude), float(longitude)


def haversine_meters(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    lat1, lon1 = first
    lat2, lon2 = second
    radius_m = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_m * c


def approximate_json_tokens(value: Any) -> int:
    return len(str(value)) // 4


def serialize_retrieval_debug(
    retrieval_result: dict[str, Any],
    *,
    limit: int = 30,
) -> dict[str, Any]:
    hints = retrieval_result["query_hints"]
    return {
        "query_hints": {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in hints.items()
        },
        "candidate_count": len(retrieval_result["ranked_rows"]),
        "context_count": len(retrieval_result["context_rows"]),
        "context_token_estimate": retrieval_result["context_token_estimate"],
        "ranked": [
            {
                "rank": row["rank"],
                "id": row["id"],
                "title": row["compact"].get("title"),
                "type": row["compact"].get("type"),
                "score": row["score"],
                "matched_concepts": sorted(row["matched_concepts"]),
                "matched_query_units": row["matched_query_units"],
                "unmatched_query_units": row["unmatched_query_units"],
                "protected_recall": row["protected_recall"],
                "signals": {
                    signal: round(value, 6)
                    for signal, value in row["signals"].items()
                },
            }
            for row in retrieval_result["ranked_rows"][:limit]
        ],
    }


def _make_scored_row(
    encoded: dict[str, Any],
    score: float,
    signals: dict[str, float],
    protected_recall: bool,
    matched_concepts: set[str],
    matched_query_units: list[str],
    unmatched_query_units: list[str],
) -> dict[str, Any]:
    return {
        "id": encoded["compact"].get("id"),
        "score": score,
        "rank": None,
        "encoded": encoded,
        "compact": encoded["compact"],
        "signals": {
            key: round(value, 6)
            for key, value in signals.items()
            if value
        },
        "protected_recall": protected_recall,
        "matched_concepts": matched_concepts,
        "matched_query_units": matched_query_units,
        "unmatched_query_units": unmatched_query_units,
    }


def _read_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _read_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]
