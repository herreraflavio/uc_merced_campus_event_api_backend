import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flask import Flask

os.environ.setdefault("OPENAI_API_KEY", "test")

from routes.ai_retrieval import rank_items  # noqa: E402
from routes import ask as ask_module  # noqa: E402
from routes import events as events_module  # noqa: E402


def item(
    item_id,
    title,
    item_type,
    tags=None,
    description="",
    **extra,
):
    return {
        "id": item_id,
        "title": title,
        "type": item_type,
        "tags": tags or [],
        "description": description,
        "subtitle": extra.pop("subtitle", ""),
        "host": extra.pop("host", "Campus Explorer"),
        "geometry": extra.pop(
            "geometry",
            {"type": "point", "latitude": 37.365, "longitude": -120.425},
        ),
        **extra,
    }


class AiRetrievalTests(unittest.TestCase):
    def test_restroom_alias_queries_surface_restroom_records(self):
        corpus = [
            item(
                "restroom-a",
                "Restroom",
                "restrooms",
                ["restrooms"],
                "Public restrooms.",
            ),
            item(
                "restroom-b",
                "Public restroom",
                "restrooms",
                ["restrooms"],
                "Public restroom open 24/7 near the library.",
            ),
            item(
                "event-a",
                "Sahara Coffee House Fundraiser",
                "event",
                ["events"],
                "Fundraising with coffee.",
            ),
            item(
                "building-a",
                "Valley Terraces Student Residences",
                "polygon",
                ["places"],
                "Suites include shared restrooms for four to six students.",
            ),
        ]

        for query in (
            "were can i go to the restroom",
            "where can i go to the restroom",
            "public restroom on campus",
            "where is a bathroom",
            "bathroom?",
            "where can I use the bathroom",
        ):
            with self.subTest(query=query):
                result = rank_items(query, corpus, max_context_items=3)
                top_ids = [row["id"] for row in result["ranked_rows"][:2]]
                context_ids = [row["id"] for row in result["context_rows"]]

                self.assertIn("restroom-a", top_ids)
                self.assertIn("restroom-b", top_ids)
                self.assertIn("restroom-a", context_ids)
                self.assertIn("restroom-b", context_ids)

    def test_structured_match_beats_deep_description_noise(self):
        corpus = [
            item(
                "restroom-a",
                "Restroom",
                "restrooms",
                ["restrooms"],
                "Public restrooms.",
            ),
            item(
                "housing-a",
                "Residence Hall",
                "polygon",
                ["places"],
                "This long building profile mentions restroom availability deep in the copy.",
            ),
            item(
                "event-a",
                "Bathroom Policy Discussion",
                "event",
                ["events"],
                "A discussion event, not a place to use the restroom.",
            ),
        ]

        result = rank_items("where is a bathroom", corpus)

        self.assertEqual(result["ranked_rows"][0]["id"], "restroom-a")
        self.assertTrue(result["ranked_rows"][0]["protected_recall"])

    def test_food_query_can_surface_event_with_descriptive_food(self):
        corpus = [
            item("dining-a", "Dining Center", "dining", ["dining"]),
            item(
                "event-food",
                "Wake Up Wednesday",
                "event",
                ["events"],
                "Weekly donuts and coffee for students.",
            ),
            item("parking-a", "Parking Lot", "parking", ["parking"]),
        ]

        result = rank_items("where can I get food", corpus, max_context_items=3)
        context_ids = [row["id"] for row in result["context_rows"]]
        food_event = next(row for row in result["ranked_rows"] if row["id"] == "event-food")

        self.assertIn("event-food", context_ids)
        self.assertIn("concept:dining", food_event["matched_query_units"])
        self.assertGreater(food_event["signals"]["query_coverage"], 0)

    def test_event_food_coverage_beats_event_only(self):
        corpus = [
            item(
                "event-food",
                "Welcome Social",
                "event",
                ["events"],
                "Join us for games and free food.",
            ),
            item(
                "event-only",
                "General Meeting",
                "event",
                ["events"],
                "Join us for a student organization meeting.",
            ),
            item("dining-a", "Dining Center", "dining", ["dining"]),
        ]

        result = rank_items("events with free food", corpus)
        top = result["ranked_rows"][0]
        event_only = next(row for row in result["ranked_rows"] if row["id"] == "event-only")

        self.assertEqual(top["id"], "event-food")
        self.assertIn("concept:event", top["matched_query_units"])
        self.assertIn("concept:dining", top["matched_query_units"])
        self.assertIn("qualifier:free", top["matched_query_units"])
        self.assertIn("qualifier:free", event_only["unmatched_query_units"])

    def test_free_parking_qualifier_beats_generic_parking(self):
        corpus = [
            item(
                "generic-parking",
                "Transportation & Parking Services",
                "parking",
                ["parking"],
                "Parking permits and campus parking information.",
            ),
            item(
                "free-parking",
                "Bellevue Lot",
                "parking",
                ["parking"],
                "Paid parking required Monday-Friday and free after 6 pm.",
            ),
        ]

        for query in (
            "where can I park for free",
            "free parking",
            "any parking that doesn't cost money",
            "where is parking free",
        ):
            with self.subTest(query=query):
                result = rank_items(query, corpus)
                top = result["ranked_rows"][0]
                generic = next(
                    row for row in result["ranked_rows"]
                    if row["id"] == "generic-parking"
                )

                self.assertEqual(top["id"], "free-parking")
                self.assertIn("qualifier:free", top["matched_query_units"])
                self.assertIn("qualifier:free", generic["unmatched_query_units"])

    def test_context_diversity_keeps_distinct_food_event(self):
        corpus = [
            item(
                f"wake-up-{index}",
                "Wake Up Wednesday",
                "event",
                ["events"],
                "Weekly donuts and coffee.",
            )
            for index in range(6)
        ]
        corpus.append(
            item(
                "recruitment-food",
                "Recruitment Day",
                "event",
                ["events"],
                "Presentation with food for attendees to enjoy.",
            )
        )

        result = rank_items("where can I get food", corpus, max_context_items=5)
        context_titles = [row["compact"]["title"] for row in result["context_rows"]]
        context_ids = [row["id"] for row in result["context_rows"]]

        self.assertLessEqual(context_titles.count("Wake Up Wednesday"), 3)
        self.assertIn("recruitment-food", context_ids)

    def test_live_corpus_mutation_requires_no_index_rebuild(self):
        corpus = [
            item("library-a", "Kolligian Library", "polygon", ["places"]),
        ]

        first = rank_items("where is a bathroom", corpus)
        self.assertNotIn("restroom-new", [row["id"] for row in first["ranked_rows"]])

        corpus.append(
            item(
                "restroom-new",
                "Public bathroom",
                "restrooms",
                ["restrooms"],
                "Public restroom near the library.",
            )
        )

        second = rank_items("where is a bathroom", corpus)
        self.assertEqual(second["ranked_rows"][0]["id"], "restroom-new")

        corpus[:] = [existing for existing in corpus if existing["id"] != "restroom-new"]
        third = rank_items("where is a bathroom", corpus)
        self.assertNotIn("restroom-new", [row["id"] for row in third["ranked_rows"]])

    def test_content_api_payload_reads_user_pages_each_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            polygons_path = tmpdir_path / "polygons.json"
            pages_path = tmpdir_path / "pages.json"
            polygons_path.write_text("[]", encoding="utf-8")
            pages_path.write_text("[]", encoding="utf-8")

            with patch.object(events_module, "POLYGONS_JSON_PATH", polygons_path), \
                patch.object(events_module, "PAGES_JSON_PATH", pages_path), \
                patch.object(events_module, "get_presence_pages_cached", return_value=[]):
                first = events_module.build_content_api_payload()
                self.assertEqual(first["pages"], [])

                pages_path.write_text(
                    json.dumps([
                        item("restroom-live", "Restroom", "restrooms", ["restrooms"])
                    ]),
                    encoding="utf-8",
                )
                second = events_module.build_content_api_payload()
                self.assertEqual(
                    [page["id"] for page in second["pages"]],
                    ["restroom-live"],
                )

                pages_path.write_text("[]", encoding="utf-8")
                third = events_module.build_content_api_payload()
                self.assertEqual(third["pages"], [])

    def test_ai_stage2_prompt_preserves_free_parking_evidence(self):
        app = Flask(__name__)
        app.register_blueprint(ask_module.ask_bp)
        corpus = [
            item(
                "free-parking",
                "Bellevue Lot",
                "polygon",
                ["parking"],
                (
                    "Bellevue Lot Green Zone Parking availability for students, "
                    "staff, faculty, visitors, and vendors. Paid parking is "
                    "required Monday - Friday from 7am - 6pm and is free after "
                    "6pm until 11:59pm."
                ),
                subtitle="Bellevue Lot Green Zone",
            ),
            item(
                "generic-parking",
                "Transportation & Parking Services",
                "polygon",
                ["parking"],
                "Transportation and parking permit information.",
            ),
        ]

        captured = {}

        def create_response(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "Bellevue Lot is free after 6pm until 11:59pm, "
                                "with paid parking required before then. "
                                "[IDS: free-parking]"
                            )
                        )
                    )
                ]
            )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create_response)
            )
        )

        with patch.object(ask_module, "build_content_api_payload", return_value={"pages": corpus}), \
            patch.object(ask_module, "client", fake_client):
            response = app.test_client().post(
                "/ai",
                json={
                    "query": "where can I park for free",
                    "item_ids": ["free-parking", "generic-parking"],
                    "debug_retrieval": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        user_prompt = captured["messages"][1]["content"]
        self.assertIn("is free after 6pm until 11:59pm", user_prompt)

        body = response.get_json()
        self.assertEqual(body["ranked_item_ids"][0], "free-parking")
        self.assertIn("free after 6pm until 11:59pm", body["ai_overview"])
        self.assertIn("free after 6pm until 11:59pm", body["citations"][0]["snippet"])
        self.assertEqual(body["stage2_debug"]["model_selected_ids"], ["free-parking"])

    def test_ai_stage2_prompt_preserves_event_food_evidence(self):
        app = Flask(__name__)
        app.register_blueprint(ask_module.ask_bp)
        corpus = [
            item(
                "event-food",
                "Welcome Social",
                "event",
                ["events"],
                "Join us Wednesday. Free food will be provided while supplies last.",
            ),
            item(
                "dining-a",
                "Campus Dining",
                "dining",
                ["dining"],
                "Campus dining location.",
            ),
        ]

        captured = {}

        def create_response(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "Welcome Social is an event with free food "
                                "provided while supplies last. [IDS: event-food]"
                            )
                        )
                    )
                ]
            )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create_response)
            )
        )

        with patch.object(ask_module, "build_content_api_payload", return_value={"pages": corpus}), \
            patch.object(ask_module, "client", fake_client):
            response = app.test_client().post(
                "/ai",
                json={
                    "query": "any events offering food",
                    "item_ids": ["event-food", "dining-a"],
                    "debug_retrieval": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        user_prompt = captured["messages"][1]["content"]
        self.assertIn("Free food will be provided while supplies last", user_prompt)

        body = response.get_json()
        self.assertEqual(body["ranked_item_ids"][0], "event-food")
        self.assertIn("while supplies last", body["citations"][0]["snippet"])
        food_candidate = next(
            candidate
            for candidate in body["stage2_debug"]["llm_candidates"]
            if candidate["id"] == "event-food"
        )
        self.assertIn("Free food will be provided", food_candidate["evidence_excerpt"])

    def test_ai_stage2_negative_answer_does_not_backfill_contradictory_results(self):
        app = Flask(__name__)
        app.register_blueprint(ask_module.ask_bp)
        corpus = [
            item(
                "free-parking",
                "Bellevue Lot",
                "polygon",
                ["parking"],
                (
                    "Paid parking is required Monday - Friday from 7am - 6pm "
                    "and is free after 6pm until 11:59pm."
                ),
            ),
            item(
                "generic-parking",
                "Transportation & Parking Services",
                "polygon",
                ["parking"],
                "Transportation and parking permit information.",
            ),
        ]

        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "The available information does not specify any "
                            "locations where parking is free."
                        )
                    )
                )
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: fake_response)
            )
        )

        with patch.object(ask_module, "build_content_api_payload", return_value={"pages": corpus}), \
            patch.object(ask_module, "client", fake_client):
            response = app.test_client().post(
                "/ai",
                json={
                    "query": "where can I park for free",
                    "item_ids": ["free-parking", "generic-parking"],
                    "debug_retrieval": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["ranked_item_ids"], [])
        self.assertEqual(body["citations"], [])
        self.assertTrue(body["stage2_debug"]["answer_claims_unavailable"])
        self.assertEqual(body["stage2_debug"]["backfilled_ids"], [])

    def test_ai_route_preserves_mobile_response_contract(self):
        app = Flask(__name__)
        app.register_blueprint(ask_module.ask_bp)
        corpus = [
            item("restroom-a", "Restroom", "restrooms", ["restrooms"]),
            item("event-a", "Game Night", "event", ["events"], "Fun games."),
        ]

        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Use the campus restroom item. [IDS: restroom-a]"
                    )
                )
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: fake_response)
            )
        )

        with patch.object(ask_module, "build_content_api_payload", return_value={"pages": corpus}), \
            patch.object(ask_module, "client", fake_client):
            response = app.test_client().post(
                "/ai",
                json={
                    "query": "where can I use the bathroom",
                    "item_ids": ["restroom-a", "event-a"],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIsInstance(body.get("ai_overview"), str)
        self.assertEqual(body["ranked_item_ids"][0], "restroom-a")
        self.assertEqual(body["citations"][0]["page_id"], "restroom-a")


if __name__ == "__main__":
    unittest.main()
