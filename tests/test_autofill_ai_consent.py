import unittest

from flask import Flask

from routes.autofill_ai import create_autofill_ai_blueprint


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("Message", (), {"content": '{"title":"Rabbit"}'})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self):
        self.completions = _FakeCompletions()
        self.chat = type("Chat", (), {"completions": self.completions})()


class AutofillAiConsentTests(unittest.TestCase):
    def setUp(self):
        self.client = _FakeClient()
        app = Flask(__name__)
        app.register_blueprint(create_autofill_ai_blueprint(self.client))
        self.http = app.test_client()

    def test_missing_consent_is_rejected_before_openai(self):
        response = self.http.post(
            "/autofill_ai",
            json={"image_urls": ["https://cdn.example.test/wildlife.jpg"]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()["error"]["code"],
            "ai_data_sharing_consent_required",
        )
        self.assertEqual(self.client.completions.calls, [])

    def test_consent_allows_openai_without_forwarding_geometry(self):
        response = self.http.post(
            "/autofill_ai",
            json={
                "ai_data_sharing_consent": True,
                "geometry": {
                    "type": "point",
                    "latitude": 37.3,
                    "longitude": -120.4,
                },
                "image_urls": ["https://cdn.example.test/wildlife.jpg"],
                "location_id": 42,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.client.completions.calls), 1)
        user_content = self.client.completions.calls[0]["messages"][1]["content"]
        context_text = user_content[0]["text"]
        self.assertNotIn("geometry", context_text)
        self.assertIn('"location_id": 42', context_text)


if __name__ == "__main__":
    unittest.main()
