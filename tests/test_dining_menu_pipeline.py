from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

from flask import Flask

ROOT_DIR = Path(__file__).resolve().parents[1]
EVENTS_PY_PATH = ROOT_DIR / "routes" / "events.py"


def load_events_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    module_name = f"events_under_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(module_name, EVENTS_PY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {EVENTS_PY_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeMenuResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeMenuSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        params = params or {}
        self.calls.append({
            "url": url,
            "params": params,
            "headers": headers,
            "timeout": timeout,
        })
        menu_group = params.get("menuGroupId", "group")
        category = params.get("categoryId", "category")
        location = params.get("locationId", "location")
        return FakeMenuResponse({
            "data": {
                "menuItems": [
                    {
                        "name": f"{location}-{menu_group}-{category}",
                        "description": "Station: Test entree",
                        "caloriesInfo": "100 Cal.",
                    }
                ]
            }
        })


class FakeScheduler:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.jobs = []
        self.running = False
        FakeScheduler.instances.append(self)

    def add_job(self, function, **kwargs):
        self.jobs.append(kwargs)

    def start(self):
        self.running = True

    def shutdown(self, wait=False):
        self.running = False


class FakeThread:
    starts = 0

    def __init__(self, target, name=None, daemon=None):
        self.target = target
        self.name = name
        self.daemon = daemon

    def start(self):
        FakeThread.starts += 1


def generated_payload():
    return {
        "PAV_nested_content": [
            {
                "title": "Sunday",
                "tabs": [
                    {
                        "title": "Breakfast",
                        "sections": [
                            {
                                "header": "Pavilion Test Pancakes",
                                "bullets": ["Description: Test item"],
                            }
                        ],
                    }
                ],
            }
        ],
        "YWDC_nested_content": [
            {
                "title": "Monday",
                "tabs": [
                    {
                        "title": "Lunch",
                        "sections": [
                            {
                                "header": "YWDC Test Bowl",
                                "bullets": ["Description: Test item"],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def write_polygons(path: Path):
    payload = [
        {
            "location_id": 774,
            "title": "The Pavilion",
            "nested_content": [
                {"title": "Amenities", "sections": [{"header": "Seating"}]},
                {
                    "title": "Sunday",
                    "tabs": [
                        {
                            "title": "Breakfast",
                            "sections": [{"header": "Old Pavilion Menu"}],
                        }
                    ],
                },
            ],
        },
        {
            "location_id": 1130,
            "title": "Yablokoff Wallace Dining Center",
            "nested_content": [
                {"title": "Hours & Info", "sections": [{"header": "Weekdays"}]},
                {
                    "title": "Monday",
                    "tabs": [
                        {
                            "title": "Lunch",
                            "sections": [{"header": "Old YWDC Menu"}],
                        }
                    ],
                },
            ],
        },
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


class DiningMenuPipelineTests(unittest.TestCase):
    def test_generator_produces_pavilion_and_ywdc_nested_content(self):
        module = load_events_module()
        fake_session = FakeMenuSession()

        module.MENU_REQUEST_DELAY_SECONDS = 0
        module.requests.Session = lambda: fake_session

        payload = module.generate_food_menu_payload()

        self.assertIn("PAV_nested_content", payload)
        self.assertIn("YWDC_nested_content", payload)
        self.assertEqual(
            [item["title"] for item in payload["PAV_nested_content"]],
            ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
        )
        self.assertEqual(
            [item["title"] for item in payload["YWDC_nested_content"]],
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        )
        self.assertGreater(module._count_menu_sections(payload["PAV_nested_content"]), 0)
        self.assertGreater(module._count_menu_sections(payload["YWDC_nested_content"]), 0)
        self.assertEqual(len(fake_session.calls), 84)

    def test_replace_preserves_non_menu_nested_content(self):
        module = load_events_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            polygons_path = Path(tmpdir) / "polygons.json"
            write_polygons(polygons_path)
            module.POLYGONS_JSON_PATH = polygons_path

            matched = module._replace_food_menu_nested_content(generated_payload())
            updated = json.loads(polygons_path.read_text(encoding="utf-8"))

        self.assertEqual(matched, {"774": 1, "1130": 1})
        pavilion = next(item for item in updated if item["location_id"] == 774)
        ywdc = next(item for item in updated if item["location_id"] == 1130)
        self.assertEqual(pavilion["nested_content"][0]["title"], "Sunday")
        self.assertEqual(pavilion["nested_content"][1]["title"], "Amenities")
        self.assertEqual(ywdc["nested_content"][0]["title"], "Monday")
        self.assertEqual(ywdc["nested_content"][1]["title"], "Hours & Info")

    def test_invalid_generated_menu_does_not_replace_existing_file(self):
        module = load_events_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            polygons_path = Path(tmpdir) / "polygons.json"
            original = write_polygons(polygons_path)
            module.POLYGONS_JSON_PATH = polygons_path
            module.generate_food_menu_payload = lambda: {
                "PAV_nested_content": [],
                "YWDC_nested_content": generated_payload()["YWDC_nested_content"],
            }

            with self.assertRaises(ValueError):
                module.generate_food_menus_and_update_polygons()

            unchanged = json.loads(polygons_path.read_text(encoding="utf-8"))

        self.assertEqual(unchanged, original)

    def test_content_api_url_contains_generated_dining_menu_data(self):
        module = load_events_module()
        app = Flask(__name__)

        with tempfile.TemporaryDirectory() as tmpdir:
            polygons_path = Path(tmpdir) / "polygons.json"
            write_polygons(polygons_path)
            module.POLYGONS_JSON_PATH = polygons_path
            module.PAGES_JSON_PATH = Path(tmpdir) / "pages.json"
            module.get_presence_pages_cached = lambda: []
            module.generate_food_menu_payload = generated_payload
            module.generate_food_menus_and_update_polygons()

            with app.test_request_context("/contentAPIURL"):
                response = module.content_api_url()
                body = response.get_json()

        pavilion = next(item for item in body["pages"] if item.get("location_id") == 774)
        ywdc = next(item for item in body["pages"] if item.get("location_id") == 1130)
        self.assertEqual(
            pavilion["nested_content"][0]["tabs"][0]["sections"][0]["header"],
            "Pavilion Test Pancakes",
        )
        self.assertEqual(
            ywdc["nested_content"][0]["tabs"][0]["sections"][0]["header"],
            "YWDC Test Bowl",
        )

    def test_successful_menu_update_invalidates_content_api_cache(self):
        module = load_events_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            polygons_path = Path(tmpdir) / "polygons.json"
            write_polygons(polygons_path)
            module.POLYGONS_JSON_PATH = polygons_path
            module.generate_food_menu_payload = generated_payload
            module.PRESENCE_PAGES_CACHE = [{"id": "stale"}]
            module.PRESENCE_PAGES_CACHE_EXPIRES_AT = (
                datetime.now(timezone.utc) + timedelta(hours=1)
            )

            module.generate_food_menus_and_update_polygons()

        self.assertIsNone(module.PRESENCE_PAGES_CACHE)
        self.assertIsNone(module.PRESENCE_PAGES_CACHE_EXPIRES_AT)

    def test_scheduler_disabled_when_flag_is_false(self):
        module = load_events_module()
        FakeScheduler.instances = []
        module.BackgroundScheduler = FakeScheduler

        with patch.dict(
            os.environ,
            {
                "RUN_STARTUP_CONTENT_PIPELINE": "false",
                "RUN_CONTENT_JOBS": "false",
            },
        ):
            scheduler = module.init_content_jobs(Flask(__name__))

        self.assertIsNone(scheduler)
        self.assertEqual(FakeScheduler.instances, [])

    def test_sunday_scheduler_is_registered_when_enabled(self):
        module = load_events_module()
        FakeScheduler.instances = []
        module.BackgroundScheduler = FakeScheduler

        with patch.dict(
            os.environ,
            {
                "RUN_STARTUP_CONTENT_PIPELINE": "false",
                "RUN_CONTENT_JOBS": "true",
            },
        ):
            scheduler = module.init_content_jobs(Flask(__name__))

        self.assertIsNotNone(scheduler)
        self.assertEqual(scheduler.kwargs["timezone"], module.PACIFIC)
        menu_job = next(
            job for job in scheduler.jobs if job["id"] == "weekly_food_menu_generation"
        )
        self.assertEqual(menu_job["trigger"], "cron")
        self.assertEqual(menu_job["day_of_week"], "sun")
        self.assertEqual(menu_job["hour"], 7)
        self.assertEqual(menu_job["minute"], 0)
        self.assertTrue(menu_job["replace_existing"])
        self.assertTrue(scheduler.running)

    def test_startup_pipeline_only_starts_once_per_process(self):
        module = load_events_module()
        module.BackgroundScheduler = None
        module.threading.Thread = FakeThread
        FakeThread.starts = 0

        with patch.dict(
            os.environ,
            {
                "RUN_STARTUP_CONTENT_PIPELINE": "true",
                "RUN_CONTENT_JOBS": "false",
            },
        ):
            module.init_content_jobs(Flask(__name__))
            module.init_content_jobs(Flask(__name__))

        self.assertEqual(FakeThread.starts, 1)
        self.assertTrue(module.STARTUP_PIPELINE_STARTED)


if __name__ == "__main__":
    unittest.main()
