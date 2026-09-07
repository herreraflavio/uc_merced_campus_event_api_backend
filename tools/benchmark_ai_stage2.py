from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OPENAI_API_KEY", "dummy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routes.ai_retrieval import normalize_text, rank_items  # noqa: E402
from routes.ask import build_ai_system_prompt, build_ai_user_content  # noqa: E402

Item = dict[str, Any]
RelevanceRule = Callable[[Item], bool]


def normalized_blob(item: Item) -> str:
    fields = [
        item.get("title"),
        item.get("subtitle"),
        item.get("description"),
        item.get("type"),
        item.get("host"),
        " ".join(item.get("tags", []) if isinstance(item.get("tags"), list) else []),
    ]
    label = item.get("label")
    if isinstance(label, dict):
        fields.append(label.get("name"))
    for key in ("nested_content", "sections"):
        nested = item.get(key)
        if isinstance(nested, (dict, list)):
            fields.append(json.dumps(nested, ensure_ascii=False))
    return normalize_text(" ".join(str(value) for value in fields if value))


def normalized_words(item: Item) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalized_blob(item)))


def has_type_or_tag(item: Item, *terms: str) -> bool:
    item_type = normalize_text(item.get("type", ""))
    tags = normalize_text(
        " ".join(item.get("tags", []) if isinstance(item.get("tags"), list) else [])
    )
    return any(term in item_type or term in tags for term in terms)


def is_event(item: Item) -> bool:
    return has_type_or_tag(item, "event")


def restroom_rule(item: Item) -> bool:
    blob = normalized_blob(item)
    return has_type_or_tag(item, "restroom", "restrooms") or "restroom" in blob


def free_parking_rule(item: Item) -> bool:
    blob = normalized_blob(item)
    words = normalized_words(item)
    return (
        has_type_or_tag(item, "parking")
        or "parking" in words
        or "park" in words
    ) and (
        "free" in words
        or "complimentary" in words
        or any(phrase in blob for phrase in ("no cost", "without cost", "no charge"))
    )


def visitor_after_hours_rule(item: Item) -> bool:
    blob = normalized_blob(item)
    words = normalized_words(item)
    return (
        ("visitor" in words or "visitor parking" in blob)
        and ("parking" in words or has_type_or_tag(item, "parking"))
        and ("after" in words or "6pm" in blob or "6 pm" in blob)
    )


def food_rule(item: Item) -> bool:
    blob = normalized_blob(item)
    return has_type_or_tag(item, "dining", "snacks") or any(
        term in blob
        for term in (
            "food",
            "dining",
            "snack",
            "vending",
            "coffee",
            "donut",
            "brunch",
            "refreshment",
            "drink",
        )
    )


def event_food_rule(item: Item) -> bool:
    return is_event(item) and food_rule(item)


def library_rule(item: Item) -> bool:
    return "library" in normalized_words(item) and not is_event(item)


def research_rule(item: Item) -> bool:
    words = normalized_words(item)
    return not is_event(item) and bool(
        words & {"research", "lab", "laboratory", "science", "engineering"}
    )


def event_game_rule(item: Item) -> bool:
    words = normalized_words(item)
    return is_event(item) and bool(words & {"game", "games", "trivia", "bingo", "movie"})


def event_rule(item: Item) -> bool:
    return is_event(item)


def wildlife_rule(item: Item) -> bool:
    return has_type_or_tag(item, "wildlife")


STAGE2_SUITE: list[dict[str, Any]] = [
    {
        "query": "where can I park for free",
        "rule": free_parking_rule,
        "evidence_terms": ("free after 6pm", "free after 6 pm", "no charge"),
    },
    {
        "query": "when is parking free",
        "rule": free_parking_rule,
        "evidence_terms": ("free after 6pm", "free after 6 pm"),
    },
    {
        "query": "visitor parking after hours",
        "rule": visitor_after_hours_rule,
        "evidence_terms": ("visitor parking", "free after 6pm", "free after 6 pm"),
    },
    {
        "query": "where can I get food",
        "rule": food_rule,
        "evidence_terms": ("food", "coffee", "donuts", "snacks", "vending"),
    },
    {
        "query": "events with food",
        "rule": event_food_rule,
        "evidence_terms": ("food", "coffee", "donuts", "brunch", "refreshments"),
    },
    {
        "query": "anything offering free food",
        "rule": event_food_rule,
        "evidence_terms": ("free food", "food", "provided"),
    },
    {
        "query": "coffee or snacks",
        "rule": food_rule,
        "evidence_terms": ("coffee", "snacks", "snack"),
    },
    {
        "query": "where can I use a restroom",
        "rule": restroom_rule,
        "evidence_terms": ("restroom", "bathroom"),
    },
    {
        "query": "public restroom",
        "rule": restroom_rule,
        "evidence_terms": ("public restroom", "restroom"),
    },
    {
        "query": "where is the library",
        "rule": library_rule,
        "evidence_terms": ("library",),
    },
    {
        "query": "which buildings are used for research",
        "rule": research_rule,
        "evidence_terms": ("research", "laboratory", "science", "engineering"),
    },
    {
        "query": "events with games",
        "rule": event_game_rule,
        "evidence_terms": ("game", "games", "bingo"),
    },
    {
        "query": "what's happening tonight",
        "rule": event_rule,
        "evidence_terms": ("event", "pm", "night", "tonight"),
    },
    {
        "query": "bird sightings",
        "rule": wildlife_rule,
        "evidence_terms": ("bird", "birds", "duck", "wildlife"),
    },
    {
        "query": "wildlife reports",
        "rule": wildlife_rule,
        "evidence_terms": ("wildlife", "sighting", "sightings"),
    },
]


def load_current_items(root: Path) -> list[Item]:
    items: list[Item] = []
    for path in (
        root / "routes" / "presence_pages_cache.json",
        root / "routes" / "polygons.json",
        root / "pages.json",
    ):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = (
            payload.get("pages")
            or payload.get("events")
            or payload.get("polygons")
            or []
        ) if isinstance(payload, dict) else payload
        items.extend(record for record in records if isinstance(record, dict))
    return items


def run_stage2_audit(root: Path, context_token_budget: int) -> dict[str, Any]:
    items = load_current_items(root)
    system_prompt = build_ai_system_prompt()
    rows = []
    latencies = []
    token_estimates = []

    for case in STAGE2_SUITE:
        query = case["query"]
        relevant_ids = {
            str(item.get("id"))
            for item in items
            if item.get("id") and case["rule"](item)
        }

        started_at = time.perf_counter()
        result = rank_items(query, items, max_context_tokens=context_token_budget)
        latency_ms = (time.perf_counter() - started_at) * 1000
        latencies.append(latency_ms)
        token_estimates.append(result["context_token_estimate"])

        candidates = result["llm_candidates"]
        candidate_ids = [str(candidate.get("id")) for candidate in candidates]
        user_prompt = build_ai_user_content(query, candidates)
        normalized_prompt = normalize_text(user_prompt)
        retained_ids = [item_id for item_id in candidate_ids if item_id in relevant_ids]
        evidence_terms = case["evidence_terms"]
        prompt_terms = [
            term
            for term in evidence_terms
            if normalize_text(term) in normalized_prompt
        ]
        first_relevant = next(
            (
                row
                for row in result["ranked_rows"]
                if str(row["id"]) in relevant_ids
            ),
            None,
        )
        first_candidate = next(
            (
                candidate
                for candidate in candidates
                if str(candidate.get("id")) in relevant_ids
            ),
            None,
        )
        first_candidate_text = normalize_text(
            json.dumps(first_candidate, ensure_ascii=False)
            if first_candidate else ""
        )
        survived_terms = [
            term
            for term in evidence_terms
            if normalize_text(term) in first_candidate_text
        ]

        rows.append({
            "query": query,
            "relevant_count": len(relevant_ids),
            "first_relevant_rank": first_relevant["rank"] if first_relevant else None,
            "first_relevant_id": first_relevant["id"] if first_relevant else None,
            "first_relevant_title": first_relevant["compact"]["title"] if first_relevant else None,
            "candidate_count": len(candidates),
            "retained_relevant_count": len(retained_ids),
            "retained_relevant_sample": retained_ids[:5],
            "evidence_survives": bool(survived_terms),
            "survived_terms": survived_terms,
            "prompt_contains_terms": prompt_terms,
            "first_candidate": first_candidate,
            "context_token_estimate": result["context_token_estimate"],
            "latency_ms": round(latency_ms, 2),
        })

    return {
        "corpus_count": len(items),
        "query_count": len(STAGE2_SUITE),
        "context_token_budget": context_token_budget,
        "system_prompt_chars": len(system_prompt),
        "averages": {
            "evidence_survival_rate": statistics.mean(
                1.0 if row["evidence_survives"] else 0.0 for row in rows
            ),
            "latency_ms": statistics.mean(latencies),
            "context_token_estimate": statistics.mean(token_estimates),
        },
        "rows": rows,
    }


def print_report(result: dict[str, Any]) -> None:
    averages = result["averages"]
    print(
        "Corpus={corpus_count} queries={query_count} context_budget={context_token_budget}".format(
            **result
        )
    )
    print(
        "evidence_survival_rate={evidence_survival_rate:.3f} "
        "avg_latency_ms={latency_ms:.2f} avg_context_tokens={context_token_estimate:.1f} "
        "system_prompt_chars={system_prompt_chars}".format(
            system_prompt_chars=result["system_prompt_chars"],
            **averages,
        )
    )
    print()
    print(
        "query | rel | first_rank | candidates | retained | evidence | ctx_tokens | first_title | terms"
    )
    print("-" * 150)
    for row in result["rows"]:
        print(
            "{query} | {relevant_count} | {first_relevant_rank} | {candidate_count} | "
            "{retained_relevant_count} | {evidence_survives} | {context_token_estimate} | "
            "{first_relevant_title} | {terms}".format(
                terms=", ".join(row["survived_terms"]),
                **row,
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Stage 2 LLM prompt/candidate evidence without calling the model."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--context-token-budget", type=int, default=4000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_stage2_audit(args.root, args.context_token_budget)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
