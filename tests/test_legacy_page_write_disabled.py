import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask

from routes import events as events_module


class LegacyPageWriteDisabledTests(unittest.TestCase):
    def test_post_add_page_is_disabled_while_content_reads_still_work(self):
        app = Flask(__name__)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pages_path = root / "pages.json"
            pages_path.write_text(
                json.dumps({"pages": [{"id": "legacy-page", "title": "Legacy"}]}),
                encoding="utf-8",
            )

            with (
                patch.object(events_module, "init_content_jobs", lambda _app: None),
                patch.object(events_module, "get_presence_pages_cached", lambda: []),
                patch.object(events_module, "PAGES_JSON_PATH", pages_path),
                patch.object(
                    events_module,
                    "POLYGONS_JSON_PATH",
                    root / "missing-polygons.json",
                ),
            ):
                app.register_blueprint(events_module.events_bp)
                client = app.test_client()

                post_response = client.post(
                    "/add/page",
                    json={"title": "Unauthenticated write must not succeed"},
                )
                read_response = client.get("/contentAPIURL")

            self.assertEqual(post_response.status_code, 404)
            self.assertEqual(read_response.status_code, 200)
            self.assertEqual(
                read_response.get_json()["pages"],
                [{"id": "legacy-page", "title": "Legacy"}],
            )
            self.assertEqual(
                json.loads(pages_path.read_text(encoding="utf-8")),
                {"pages": [{"id": "legacy-page", "title": "Legacy"}]},
            )


if __name__ == "__main__":
    unittest.main()
