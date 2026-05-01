# test_llm.py

import os
import json
import time
import math
import copy
import requests
import statistics
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from matplotlib.lines import Line2D


# ============================================================
# CONFIG
# ============================================================

BASE_URL = os.getenv("AI_BASE_URL", "http://10.56.184.54:8080")

CONTENT_API_URL = os.getenv(
    "CONTENT_API_URL",
    f"{BASE_URL}/contentAPIURL"
)

OUTPUT_DIR = Path("ai_metrics_output")
OUTPUT_DIR.mkdir(exist_ok=True)

METRICS_JSON_FILE = OUTPUT_DIR / "ai_pipeline_metrics.json"
SUMMARY_JSON_FILE = OUTPUT_DIR / "ai_pipeline_summary.json"
REQUESTS_JSON_FILE = OUTPUT_DIR / "ai_pipeline_requests_responses.json"

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))

AUTO_FILL_EMPTY_ITEM_IDS = True

PIPELINES = [
    {
        "key": "baseline",
        "label": "Baseline GPT-4o Full Context",
        "url": f"{BASE_URL}/ai_baseline",
    },
    {
        "key": "enhanced",
        "label": "Enhanced Ranked Full Context + GPT-4o",
        "url": f"{BASE_URL}/ai_enhanced",
    },
    {
        "key": "baseline_reasoning",
        "label": "Baseline GPT-5 Reasoning Full Context",
        "url": f"{BASE_URL}/ask_baseline_reasoning",
    },
]


# ============================================================
# EVALUATION CASES
# ============================================================

EVAL_CASES = [
    {
        "id": "events_1",
        "category": "events",
        "query": "events offering food or drinks",
        "item_ids": [],
        "expected_item_ids": [
            "fcbbfb16-332d-42a7-97e2-195b80acb918",
            "64606d38-8948-4294-8211-b074ab6f85b9",
            "bbc855ff-5274-4f5d-9413-a3aa9b9b8da7"
        ],
    },
    {
        "id": "events_2",
        "category": "events",
        "query": "any events that will do games",
        "item_ids": [],
        "expected_item_ids": [
            "9e5067f9-7724-46b7-9870-4989ebb656e3",
      
            
         
        ],
    },
    {
        "id": "dining_1",
        "category": "dining",
        "query": "where can I get food near me",
        "item_ids": [],
        "expected_item_ids": [
            "polygon1761862422686169",
            "polygon1761861841336199"
        ],
    },
    {
        "id": "dining_2",
        "category": "dining",
        "query": "coffee or snacks on campus",
        "item_ids": [],
        "expected_item_ids": [
            "A013455F-58BC-46F5-A418-308AE55563F0",
       
            "B6F4A730-8158-415B-AF6F-EA9954C30281"
        ],
    },
    {
        "id": "buildings_locations_1",
        "category": "buildings / locations",
        "query": "where is the library",
        "item_ids": [],
        "expected_item_ids": [
            "polygon1761861346599768"
        ],
    },
    {
        "id": "buildings_locations_2",
        "category": "buildings / locations",
        "query": "what buildings do research",
        "item_ids": [],
        "expected_item_ids": [
            "polygon1761861787851598",
            "polygon176186469807932",
            "polygon1761860891689236"
        ],
    },
    {
        "id": "services_1",
        "category": "services",
        "query": "what services does uc merced offer",
        "item_ids": [],
        "expected_item_ids": [
        "polygon1761858712943575",
  "polygon1761860370535493",
  "polygon1761860435344370",
  "polygon1761862947648773",
  "polygon1761861907489943",
  "polygon1761863691369604",
  "polygon1761863882489923",
  "polygon1761864638825198",
  "polygon1763160787041855"
        ],
    },
    {
        "id": "services_2",
        "category": "services",
        "query": "public restroom on campus",
        "item_ids": [],
        "expected_item_ids": [
            "21E8F99C-9DC8-4088-8203-58CA3E640156",
            "2E1E4CC7-FCCC-47B0-9D6C-138CA003C306"
            "2E1E4CC7-FCCC-47B0-9D6C-138CA003C306"
        ],
    },
    {
        "id": "user_generated_1",
        "category": "user-generated data",
        "query": "wildlife sightings near campus",
        "item_ids": [],
        "expected_item_ids": [
            "CBA13787-EFCC-4256-8608-B0DEB1E8EE1F",
            "55C99F32-F81F-483C-8F9B-383F971BBBF6"
        ],
    },
    {
        "id": "user_generated_2",
        "category": "user-generated data",
        "query": "any sightings of birds",
        "item_ids": [],
        "expected_item_ids": [
            "55C99F32-F81F-483C-8F9B-383F971BBBF6"
        ],
    },
]


# ============================================================
# HELPERS
# ============================================================

def fetch_all_content_item_ids():
    try:
        resp = requests.get(CONTENT_API_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()

        data = resp.json()
        pages = data.get("pages", [])

        if not isinstance(pages, list):
            print("WARNING: contentAPIURL response does not contain pages array.")
            return []

        return [
            item.get("id")
            for item in pages
            if item.get("id")
        ]

    except Exception as e:
        print(f"WARNING: Could not fetch content item IDs: {e}")
        return []


def build_request_payload(case, all_content_item_ids):
    case_item_ids = case.get("item_ids", [])

    if AUTO_FILL_EMPTY_ITEM_IDS and not case_item_ids:
        item_ids = list(all_content_item_ids)
    else:
        item_ids = list(case_item_ids)

    return {
        "item_ids": item_ids,
        "query": case["query"],
    }


def safe_mean(values):
    valid = [
        v for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]

    if not valid:
        return None

    return sum(valid) / len(valid)


def clean_usage(usage):
    if not isinstance(usage, dict):
        usage = {}

    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    reasoning_tokens = usage.get("reasoning_tokens", 0) or 0
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens) or 0

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "context_tokens": usage.get("context_tokens", input_tokens) or 0,
        "total_tokens": total_tokens,
    }


def extract_api_cost(response_json):
    if not isinstance(response_json, dict):
        return 0

    cost = response_json.get("cost", {})

    if isinstance(cost, dict):
        total = cost.get("total_cost_usd")

        if total is not None:
            try:
                return round(float(total), 8)
            except (TypeError, ValueError):
                pass

    total = response_json.get("estimated_cost_usd")

    if total is not None:
        try:
            return round(float(total), 8)
        except (TypeError, ValueError):
            pass

    return 0


def score_ranking(ranked_item_ids, expected_item_ids):
    expected = set(expected_item_ids or [])

    if not expected:
        return {
            "has_ground_truth": False,
            "top_1_accuracy": None,
            "top_3_accuracy": None,
            "relevance_score": None,
            "matched_expected_ids": [],
        }

    ranked = ranked_item_ids or []

    top_1 = ranked[:1]
    top_3 = ranked[:3]

    top_1_accuracy = 1 if any(item_id in expected for item_id in top_1) else 0
    top_3_accuracy = 1 if any(item_id in expected for item_id in top_3) else 0

    weights = [1.0, 0.66, 0.33]

    raw_score = 0
    matched_expected_ids = []

    for index, item_id in enumerate(top_3):
        if item_id in expected:
            raw_score += weights[index]
            matched_expected_ids.append(item_id)

    max_possible_items = min(len(expected), 3)
    max_possible_score = sum(weights[:max_possible_items])

    relevance_score = raw_score / max_possible_score if max_possible_score else None

    return {
        "has_ground_truth": True,
        "top_1_accuracy": top_1_accuracy,
        "top_3_accuracy": top_3_accuracy,
        "relevance_score": round(relevance_score, 4) if relevance_score is not None else None,
        "matched_expected_ids": matched_expected_ids,
    }


def call_pipeline(pipeline, case, all_content_item_ids):
    payload = build_request_payload(
        case=case,
        all_content_item_ids=all_content_item_ids
    )

    request_payload = copy.deepcopy(payload)

    request_record = {
        "case_id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "pipeline_key": pipeline["key"],
        "pipeline_label": pipeline["label"],
        "endpoint": pipeline["url"],
        "request_payload": request_payload,
        "response_json": None,
        "response_text": None,
        "status_code": None,
        "success": False,
        "latency_seconds": None,
        "error": None,
    }

    started = time.perf_counter()

    try:
        resp = requests.post(
            pipeline["url"],
            json=copy.deepcopy(payload),
            timeout=REQUEST_TIMEOUT_SECONDS
        )

        latency_seconds = time.perf_counter() - started
        status_code = resp.status_code

        request_record["status_code"] = status_code
        request_record["success"] = 200 <= status_code < 300
        request_record["latency_seconds"] = round(latency_seconds, 4)

        try:
            response_json = resp.json()
            request_record["response_json"] = response_json
        except Exception:
            response_json = {
                "raw_text": resp.text
            }
            request_record["response_json"] = response_json
            request_record["response_text"] = resp.text

        ranked_item_ids = response_json.get("ranked_item_ids", [])

        if not isinstance(ranked_item_ids, list):
            ranked_item_ids = []

        usage_clean = clean_usage(response_json.get("usage", {}))
        cost_usd = extract_api_cost(response_json)

        ranking_scores = score_ranking(
            ranked_item_ids=ranked_item_ids,
            expected_item_ids=case.get("expected_item_ids", [])
        )

        return {
            "case_id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "pipeline_key": pipeline["key"],
            "pipeline_label": pipeline["label"],
            "endpoint": pipeline["url"],
            "status_code": status_code,
            "success": 200 <= status_code < 300,
            "latency_seconds": round(latency_seconds, 4),
            "cost_usd": cost_usd,
            "estimated_cost_usd": cost_usd,
            "usage": usage_clean,
            "model": response_json.get("model"),
            "reasoning_effort": response_json.get("reasoning_effort"),
            "api_cost_detail": response_json.get("cost", {}),
            "ranked_item_ids": ranked_item_ids,
            "expected_item_ids": case.get("expected_item_ids", []),
            "scores": ranking_scores,
            "ai_overview": response_json.get("ai_overview", ""),
            "citations": response_json.get("citations", []),
            "error": response_json.get("error"),
            "details": response_json.get("details"),
            "request_payload": request_payload,
            "response_json": response_json,
            "request_response_record": request_record,
        }

    except Exception as e:
        latency_seconds = time.perf_counter() - started

        request_record["success"] = False
        request_record["latency_seconds"] = round(latency_seconds, 4)
        request_record["error"] = str(e)

        return {
            "case_id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "pipeline_key": pipeline["key"],
            "pipeline_label": pipeline["label"],
            "endpoint": pipeline["url"],
            "status_code": None,
            "success": False,
            "latency_seconds": round(latency_seconds, 4),
            "cost_usd": 0,
            "estimated_cost_usd": 0,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "context_tokens": 0,
                "total_tokens": 0,
            },
            "model": None,
            "reasoning_effort": None,
            "api_cost_detail": {},
            "ranked_item_ids": [],
            "expected_item_ids": case.get("expected_item_ids", []),
            "scores": {
                "has_ground_truth": bool(case.get("expected_item_ids")),
                "top_1_accuracy": None,
                "top_3_accuracy": None,
                "relevance_score": None,
                "matched_expected_ids": [],
            },
            "ai_overview": "",
            "citations": [],
            "error": str(e),
            "details": None,
            "request_payload": request_payload,
            "response_json": None,
            "request_response_record": request_record,
        }


def build_summary(results):
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_cases": len(EVAL_CASES),
        "total_pipeline_requests": len(results),
        "pipelines": {},
        "by_category": {},
    }

    for pipeline in PIPELINES:
        key = pipeline["key"]
        rows = [r for r in results if r["pipeline_key"] == key]

        top_1_values = [r["scores"]["top_1_accuracy"] for r in rows]
        top_3_values = [r["scores"]["top_3_accuracy"] for r in rows]
        relevance_values = [r["scores"]["relevance_score"] for r in rows]

        latencies = [
            r["latency_seconds"]
            for r in rows
            if r["latency_seconds"] is not None
        ]

        costs = [r["cost_usd"] for r in rows]
        context_tokens = [r["usage"]["context_tokens"] for r in rows]
        input_tokens = [r["usage"]["input_tokens"] for r in rows]
        output_tokens = [r["usage"]["output_tokens"] for r in rows]
        reasoning_tokens = [r["usage"]["reasoning_tokens"] for r in rows]
        total_tokens = [r["usage"]["total_tokens"] for r in rows]

        summary["pipelines"][key] = {
            "label": pipeline["label"],
            "endpoint": pipeline["url"],
            "success_count": sum(1 for r in rows if r["success"]),
            "error_count": sum(1 for r in rows if not r["success"]),
            "top_1_accuracy": safe_mean(top_1_values),
            "top_3_accuracy": safe_mean(top_3_values),
            "average_relevance_score": safe_mean(relevance_values),
            "average_latency_seconds": safe_mean(latencies),
            "median_latency_seconds": statistics.median(latencies) if latencies else None,
            "average_cost_usd": safe_mean(costs),
            "average_estimated_cost_usd": safe_mean(costs),
            "total_cost_usd": round(sum(costs), 8),
            "total_estimated_cost_usd": round(sum(costs), 8),
            "average_context_tokens": safe_mean(context_tokens),
            "average_input_tokens": safe_mean(input_tokens),
            "average_output_tokens": safe_mean(output_tokens),
            "average_reasoning_tokens": safe_mean(reasoning_tokens),
            "average_total_tokens": safe_mean(total_tokens),
        }

    categories = sorted(set(case["category"] for case in EVAL_CASES))

    for category in categories:
        summary["by_category"][category] = {}

        for pipeline in PIPELINES:
            key = pipeline["key"]
            rows = [
                r for r in results
                if r["category"] == category and r["pipeline_key"] == key
            ]

            relevance_values = [r["scores"]["relevance_score"] for r in rows]
            top_1_values = [r["scores"]["top_1_accuracy"] for r in rows]
            top_3_values = [r["scores"]["top_3_accuracy"] for r in rows]
            latencies = [
                r["latency_seconds"]
                for r in rows
                if r["latency_seconds"] is not None
            ]
            costs = [r["cost_usd"] for r in rows]

            summary["by_category"][category][key] = {
                "label": pipeline["label"],
                "endpoint": pipeline["url"],
                "average_relevance_score": safe_mean(relevance_values),
                "top_1_accuracy": safe_mean(top_1_values),
                "top_3_accuracy": safe_mean(top_3_values),
                "average_latency_seconds": safe_mean(latencies),
                "average_cost_usd": safe_mean(costs),
            }

    return summary


def build_requests_response_export(results):
    return {
        "generated_at": datetime.now().isoformat(),
        "total_requests": len(results),
        "requests": [
            result["request_response_record"]
            for result in results
        ],
    }


# ============================================================
# GRAPHING
# ============================================================

def value_or_zero(value):
    return value if value is not None else 0


def plot_source_ranking_quality(summary):
    labels = [pipeline["label"] for pipeline in PIPELINES]

    top_1 = [
        value_or_zero(summary["pipelines"][pipeline["key"]]["top_1_accuracy"])
        for pipeline in PIPELINES
    ]

    top_3 = [
        value_or_zero(summary["pipelines"][pipeline["key"]]["top_3_accuracy"])
        for pipeline in PIPELINES
    ]

    x = range(len(labels))
    width = 0.25

    plt.figure(figsize=(11, 6))

    plt.bar([i - width / 2 for i in x], top_1, width, label="Top-1 Accuracy")
    plt.bar([i + width / 2 for i in x], top_3, width, label="Top-3 Accuracy")

    plt.ylabel("Accuracy")
    plt.xlabel("Pipeline")
    plt.title("Source Ranking Quality by Pipeline")
    plt.xticks(list(x), labels, rotation=12, ha="right")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "source_ranking_quality_by_pipeline.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def plot_answer_relevance_by_query_type(summary):
    categories = list(summary["by_category"].keys())
    x = range(len(categories))
    width = 0.25

    plt.figure(figsize=(13, 6))

    for offset_index, pipeline in enumerate(PIPELINES):
        key = pipeline["key"]

        values = [
            value_or_zero(
                summary["by_category"][category][key]["average_relevance_score"]
            )
            for category in categories
        ]

        offset = (offset_index - 1) * width

        plt.bar(
            [i + offset for i in x],
            values,
            width,
            label=pipeline["label"]
        )

    plt.ylabel("Average Relevance Score")
    plt.xlabel("Query Type")
    plt.title("Answer Relevance Across Query Types")
    plt.xticks(list(x), categories, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "answer_relevance_across_query_types.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def plot_cost_latency_context_tradeoff(summary):
    plt.figure(figsize=(10, 6))

    max_latency = 0
    legend_labels = []

    for pipeline in PIPELINES:
        key = pipeline["key"]
        row = summary["pipelines"][key]

        avg_latency = value_or_zero(row["average_latency_seconds"])
        max_latency = max(max_latency, avg_latency)

        avg_quality = value_or_zero(row["average_relevance_score"])
        avg_context_tokens = value_or_zero(row["average_context_tokens"])

        bubble_size = max(100, avg_context_tokens / 10)

        label = (
            f"{pipeline['label']}\n"
            f"Cost/run: ${value_or_zero(row['average_cost_usd']):.6f}"
        )
        legend_labels.append(label)

        plt.scatter(
            avg_latency,
            avg_quality,
            s=bubble_size,
            alpha=0.65
        )

        plt.annotate(
            pipeline["label"],
            (avg_latency, avg_quality),
            textcoords="offset points",
            xytext=(8, 8),
            ha="left"
        )

    plt.xlim(left=0, right=max_latency * 1.35 if max_latency > 0 else 1)

    plt.ylabel("Answer Quality / Relevance Score")
    plt.xlabel("Average Response Latency Seconds")
    plt.title("Cost–Latency–Context Trade-off")
    plt.ylim(0, 1.05)

    # Text-only legend (no bubble markers shown)
    legend_handles = [
        Line2D([], [], linestyle="None")
        for _ in legend_labels
    ]

    plt.legend(
        handles=legend_handles,
        labels=legend_labels,
        handlelength=0,
        handletextpad=0,
        frameon=True
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "cost_latency_context_tradeoff.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path
# ============================================================
# MAIN
# ============================================================

def main():
    print("Fetching candidate content item IDs...")
    all_content_item_ids = fetch_all_content_item_ids()

    if AUTO_FILL_EMPTY_ITEM_IDS:
        print(f"Loaded {len(all_content_item_ids)} content item IDs.")

    results = []

    total_requests = len(EVAL_CASES) * len(PIPELINES)
    request_count = 0

    for case in EVAL_CASES:
        print(f"\nCASE: {case['id']} | {case['category']} | {case['query']}")

        for pipeline in PIPELINES:
            request_count += 1

            print(
                f"  [{request_count}/{total_requests}] "
                f"Calling {pipeline['label']}..."
            )

            result = call_pipeline(
                pipeline=pipeline,
                case=case,
                all_content_item_ids=all_content_item_ids
            )

            results.append(result)
            scores = result["scores"]

            model_part = f" model={result['model']}" if result.get("model") else ""
            reasoning_part = (
                f" reasoning={result['reasoning_effort']}"
                if result.get("reasoning_effort")
                else ""
            )

            print(
                f"    status={result['status_code']} "
                f"success={result['success']} "
                f"latency={result['latency_seconds']}s "
                f"top1={scores['top_1_accuracy']} "
                f"top3={scores['top_3_accuracy']} "
                f"relevance={scores['relevance_score']} "
                f"context_tokens={result['usage']['context_tokens']} "
                f"input_tokens={result['usage']['input_tokens']} "
                f"output_tokens={result['usage']['output_tokens']} "
                f"reasoning_tokens={result['usage']['reasoning_tokens']} "
                f"cost=${result['cost_usd']}"
                f"{model_part}"
                f"{reasoning_part}"
            )

    summary = build_summary(results)
    requests_response_export = build_requests_response_export(results)

    with open(METRICS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(SUMMARY_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(REQUESTS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(requests_response_export, f, indent=2)

    graph_1 = plot_source_ranking_quality(summary)
    graph_2 = plot_answer_relevance_by_query_type(summary)
    graph_3 = plot_cost_latency_context_tradeoff(summary)

    print("\nDone.")
    print(f"Saved raw metrics:        {METRICS_JSON_FILE}")
    print(f"Saved summary:            {SUMMARY_JSON_FILE}")
    print(f"Saved requests/responses: {REQUESTS_JSON_FILE}")
    print(f"Saved graph 1:            {graph_1}")
    print(f"Saved graph 2:            {graph_2}")
    print(f"Saved graph 3:            {graph_3}")

    print("\nPipeline Summary:")

    for pipeline in PIPELINES:
        key = pipeline["key"]
        row = summary["pipelines"][key]

        print(f"\n{row['label']}")
        print(f"  Top-1 Accuracy:          {row['top_1_accuracy']}")
        print(f"  Top-3 Accuracy:          {row['top_3_accuracy']}")
        print(f"  Avg Relevance Score:     {row['average_relevance_score']}")
        print(f"  Avg Latency Seconds:     {row['average_latency_seconds']}")
        print(f"  Avg Context Tokens:      {row['average_context_tokens']}")
        print(f"  Avg Input Tokens:        {row['average_input_tokens']}")
        print(f"  Avg Output Tokens:       {row['average_output_tokens']}")
        print(f"  Avg Reasoning Tokens:    {row['average_reasoning_tokens']}")
        print(f"  Avg Total Tokens:        {row['average_total_tokens']}")
        print(f"  Avg Cost:                ${row['average_cost_usd']}")
        print(f"  Total Cost:              ${row['total_cost_usd']}")


if __name__ == "__main__":
    main()