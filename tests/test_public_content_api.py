import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask

from routes import events as events_module


class PublicContentApiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.polygons_path = self.root / "polygons.json"
        self.pages_path = self.root / "pages.json"
        self.polygons = [{"id": "polygon-library", "title": "Library"}]
        self.events = [{"id": "presence-event", "type": "event"}]
        self.user_pages = [
            {"id": "wildlife-post", "type": "wildlife"},
            {"id": "restroom-point", "type": "restrooms"},
        ]
        self.pages_path.write_text(
            json.dumps({"pages": self.user_pages}), encoding="utf-8"
        )
        self.enterContext(patch.object(events_module, "init_content_jobs"))
        self.enterContext(
            patch.object(events_module, "POLYGONS_JSON_PATH", self.polygons_path)
        )
        self.pages_source = self.enterContext(
            patch.object(events_module, "PAGES_JSON_PATH", wraps=self.pages_path)
        )
        self.presence = self.enterContext(
            patch.object(
                events_module, "get_presence_pages_cached", return_value=self.events
            )
        )
        app = Flask(__name__)
        app.register_blueprint(events_module.events_bp)
        self.client = app.test_client()

    def test_public_content_returns_only_polygons_in_both_file_shapes(self):
        for payload in (self.polygons, {"polygons": self.polygons}):
            with self.subTest(payload=payload):
                self.polygons_path.write_text(json.dumps(payload), encoding="utf-8")
                response = self.client.get("/publicContentAPIURL")

                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.is_json)
                self.assertEqual(response.get_json(), {"pages": self.polygons})
                self.presence.assert_not_called()
                self.pages_source.exists.assert_not_called()
                self.pages_source.open.assert_not_called()

    def test_existing_content_endpoint_still_includes_all_sources(self):
        self.polygons_path.write_text(json.dumps(self.polygons), encoding="utf-8")

        response = self.client.get("/contentAPIURL")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {"pages": self.events + self.polygons + self.user_pages},
        )

    def test_missing_polygons_returns_empty_pages(self):
        response = self.client.get("/publicContentAPIURL")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"pages": []})
        self.presence.assert_not_called()
        self.pages_source.open.assert_not_called()

    def test_invalid_polygons_logs_error_and_keeps_response_shape(self):
        for content in ("not json", '{"polygons": "invalid"}'):
            with self.subTest(content=content):
                self.polygons_path.write_text(content, encoding="utf-8")
                with self.assertLogs(events_module.logger, level="ERROR") as logs:
                    response = self.client.get("/publicContentAPIURL")

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json(), {"pages": []})
                self.assertIn("PublicContentAPI Pipeline Error (Polygons)", logs.output[0])
                self.presence.assert_not_called()
                self.pages_source.open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
