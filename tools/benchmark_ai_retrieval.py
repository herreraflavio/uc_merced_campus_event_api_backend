from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routes.ai_retrieval import normalize_text, rank_items  # noqa: E402

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
    item_type = str(item.get("type", "")).lower()
    tags = item.get("tags", [])
    tag_values = {
        str(tag).lower()
        for tag in tags
        if isinstance(tag, str)
    }
    return any(term in item_type or term in tag_values for term in terms)


def restroom_rule(item: Item) -> bool:
    blob = normalized_blob(item)
    return has_type_or_tag(item, "restroom", "restrooms") or "restroom" in blob


def library_rule(item: Item) -> bool:
    blob = normalized_blob(item)
    return "library" in blob and not has_type_or_tag(item, "event")


def research_building_rule(item: Item) -> bool:
    blob = normalized_blob(item)
    return (
        not has_type_or_tag(item, "event")
        and any(term in blob for term in ("research", "lab", "laboratory", "science", "engineering"))
    )


def dining_rule(item: Item) -> bool:
    blob = normalized_blob(item)
    return (
        has_type_or_tag(item, "dining", "snacks", "food")
        or any(term in blob for term in ("food", "dining", "coffee", "snack", "vending", "restaurant"))
    )


def event_rule(item: Item) -> bool:
    return has_type_or_tag(item, "event", "events")


def event_food_rule(item: Item) -> bool:
    blob = normalized_blob(item)
    return event_rule(item) and any(
        term in blob
        for term in (
            "food",
            "drink",
            "coffee",
            "donut",
            "brunch",
            "snack",
            "refreshment",
            "provided",
            "serving",
        )
    )


def event_game_rule(item: Item) -> bool:
    blob = normalized_blob(item)
    return event_rule(item) and any(
        term in blob for term in ("game", "games", "trivia", "bingo", "movie", "social")
    )


def wildlife_rule(item: Item) -> bool:
    return has_type_or_tag(item, "wildlife")


def food_or_refreshment_rule(item: Item) -> bool:
    blob = normalized_blob(item)
    return (
        dining_rule(item)
        or any(
            term in blob
            for term in (
                "food",
                "drink",
                "coffee",
                "donut",
                "brunch",
                "snack",
                "refreshment",
            )
        )
    )


def free_parking_rule(item: Item) -> bool:
    blob = normalized_blob(item)
    words = normalized_words(item)
    has_parking = (
        has_type_or_tag(item, "parking")
        or "parking" in words
        or "park" in words
    )
    has_free = (
        "free" in words
        or "complimentary" in words
        or any(phrase in blob for phrase in ("no cost", "without cost", "no charge"))
    )
    return has_parking and has_free


QUERY_SUITE: list[tuple[str, RelevanceRule]] = [
    ("public restroom on campus", restroom_rule),
    ("where can I go to the restroom", restroom_rule),
    ("were can i go to the restroom", restroom_rule),
    ("where is a bathroom", restroom_rule),
    ("bathroom?", restroom_rule),
    ("where can I use the bathroom", restroom_rule),
    ("where is the library", library_rule),
    ("were is the library", library_rule),
    ("what buildings do research", research_building_rule),
    ("where can I get food", dining_rule),
    ("any food on campus", food_or_refreshment_rule),
    ("anywhere i can grab food", dining_rule),
    ("coffee or snacks on campus", dining_rule),
    ("anything offering food", food_or_refreshment_rule),
    ("anywhere giving out food", food_or_refreshment_rule),
    ("events offering food or drinks", event_food_rule),
    ("events with food", event_food_rule),
    ("events with coffee snacks", event_food_rule),
    ("events that will do games", event_game_rule),
    ("events with games", event_game_rule),
    ("anything fun going on", event_rule),
    ("things happening tonight", event_rule),
    ("free parking", free_parking_rule),
    ("where can I park for free", free_parking_rule),
    ("parking that doesn't cost money", free_parking_rule),
    ("where is parking free", free_parking_rule),
    ("wildlife sightings near campus", wildlife_rule),
    ("any sightings of birds", wildlife_rule),
    ("seen any birds around campus", wildlife_rule),
]


def load_current_items(root: Path) -> list[Item]:
    sources = [
        root / "routes" / "presence_pages_cache.json",
        root / "routes" / "polygons.json",
        root / "pages.json",
    ]
    items: list[Item] = []

    for path in sources:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            records = (
                payload.get("pages")
                or payload.get("events")
                or payload.get("polygons")
                or []
            )
        else:
            records = payload
        items.extend(record for record in records if isinstance(record, dict))

    return items


def calculate_metrics(
    ranked_ids: list[str],
    relevant_ids: set[str],
    *,
    recall_k: int,
) -> dict[str, float]:
    if not relevant_ids:
        return {
            "top1": 0.0,
            "top3": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
        }

    first_relevant_rank = next(
        (index + 1 for index, item_id in enumerate(ranked_ids) if item_id in relevant_ids),
        None,
    )
    top_k_ids = set(ranked_ids[:recall_k])

    return {
        "top1": 1.0 if ranked_ids[:1] and ranked_ids[0] in relevant_ids else 0.0,
        "top3": 1.0 if any(item_id in relevant_ids for item_id in ranked_ids[:3]) else 0.0,
        "recall": len(top_k_ids & relevant_ids) / len(relevant_ids),
        "mrr": 0.0 if first_relevant_rank is None else 1 / first_relevant_rank,
    }


def run_benchmark(
    root: Path,
    recall_k: int,
    context_token_budget: int | None,
) -> dict[str, Any]:
    items = load_current_items(root)
    rows = []
    latencies = []
    token_estimates = []

    for query, relevance_rule in QUERY_SUITE:
        relevant_ids = {
            str(item.get("id"))
            for item in items
            if item.get("id") and relevance_rule(item)
        }

        started_at = time.perf_counter()
        result = rank_items(
            query,
            items,
            max_context_tokens=context_token_budget,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        latencies.append(latency_ms)
        token_estimates.append(result["context_token_estimate"])

        ranked_ids = [str(row["id"]) for row in result["ranked_rows"]]
        metrics = calculate_metrics(ranked_ids, relevant_ids, recall_k=recall_k)
        first_relevant = next(
            (row for row in result["ranked_rows"] if str(row["id"]) in relevant_ids),
            None,
        )
        rows.append({
            "query": query,
            "relevant_count": len(relevant_ids),
            "first_relevant_rank": first_relevant["rank"] if first_relevant else None,
            "first_relevant_id": first_relevant["id"] if first_relevant else None,
            "first_relevant_title": first_relevant["compact"]["title"] if first_relevant else None,
            "first_relevant_signals": first_relevant["signals"] if first_relevant else {},
            "first_relevant_matched_query_units": (
                first_relevant["matched_query_units"] if first_relevant else []
            ),
            "first_relevant_unmatched_query_units": (
                first_relevant["unmatched_query_units"] if first_relevant else []
            ),
            "context_count": len(result["context_rows"]),
            "context_token_estimate": result["context_token_estimate"],
            "latency_ms": round(latency_ms, 2),
            **metrics,
        })

    return {
        "corpus_count": len(items),
        "query_count": len(QUERY_SUITE),
        "recall_k": recall_k,
        "context_token_budget": context_token_budget,
        "averages": {
            "top1": statistics.mean(row["top1"] for row in rows),
            "top3": statistics.mean(row["top3"] for row in rows),
            "recall": statistics.mean(row["recall"] for row in rows),
            "mrr": statistics.mean(row["mrr"] for row in rows),
            "latency_ms": statistics.mean(latencies),
            "context_token_estimate": statistics.mean(token_estimates),
        },
        "rows": rows,
    }


def print_report(result: dict[str, Any]) -> None:
    averages = result["averages"]
    print(
        "Corpus={corpus_count} queries={query_count} Recall@{recall_k}".format(
            **result
        )
    )
    print(
        "avg_top1={top1:.3f} avg_top3={top3:.3f} avg_recall={recall:.3f} "
        "avg_mrr={mrr:.3f} avg_latency_ms={latency_ms:.2f} "
        "avg_context_tokens={context_token_estimate:.1f}".format(**averages)
    )
    print()
    print(
        "query | rel | first_rank | top1 | top3 | recall | mrr | ms | "
        "ctx_tokens | coverage | bm25 | first_title"
    )
    print("-" * 150)
    for row in result["rows"]:
        signals = row["first_relevant_signals"]
        print(
            "{query} | {relevant_count} | {first_relevant_rank} | "
            "{top1:.0f} | {top3:.0f} | {recall:.3f} | {mrr:.3f} | "
            "{latency_ms:.2f} | {context_token_estimate} | {coverage} | "
            "{bm25} | {first_relevant_title}".format(
                coverage=signals.get("query_coverage", ""),
                bm25=signals.get("bm25", ""),
                **row,
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark live AI retrieval quality.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--recall-k", type=int, default=10)
    parser.add_argument("--context-token-budget", type=int, default=4000)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    result = run_benchmark(
        args.root,
        args.recall_k,
        args.context_token_budget,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
