import logging
import unittest

from app.home_location import (
    HomeLocation,
    append_home_location_context,
    derive_ha_config_url,
    fetch_home_location,
    load_home_location_if_enabled,
    query_needs_coordinates,
)


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size: int) -> bytes:
        return self.body[:size]


class HomeLocationTests(unittest.TestCase):
    def test_sanitizes_and_validates_home_assistant_config(self):
        location = HomeLocation.from_ha_config(
            {
                "location_name": "  Los\nAngeles\u202e  ",
                "country": "us",
                "time_zone": "America/Los_Angeles",
                "latitude": "33.864321",
                "longitude": -118.399876,
            }
        )

        self.assertIsNotNone(location)
        assert location is not None
        self.assertEqual(location.location_name, "Los Angeles")
        self.assertEqual(location.country, "US")
        self.assertEqual(location.timezone, "America/Los_Angeles")
        self.assertEqual(location.rounded_coordinates(), (33.86, -118.4))
        self.assertEqual(location.city_for_search(), "Los Angeles")

    def test_invalid_fields_are_dropped_and_coordinate_pair_stays_atomic(self):
        location = HomeLocation.from_ha_config(
            {
                "location_name": "Home",
                "country": "USA",
                "time_zone": "../../not-a-zone",
                "latitude": 91,
                "longitude": -118.4,
            }
        )

        self.assertEqual(location, HomeLocation(location_name="Home"))
        assert location is not None
        self.assertFalse(location.has_coordinates)
        self.assertIsNone(location.web_search_user_location())

    def test_exact_coordinates_cannot_leak_through_repr_or_safe_summary(self):
        location = HomeLocation.from_ha_config(
            {
                "location_name": "Home",
                "country": "US",
                "time_zone": "America/Los_Angeles",
                "latitude": 33.864321,
                "longitude": -118.399876,
            }
        )
        assert location is not None

        for safe_text in (repr(location), location.safe_summary()):
            self.assertNotIn("33.864321", safe_text)
            self.assertNotIn("-118.399876", safe_text)

    def test_instance_name_is_not_blindly_passed_as_city(self):
        location = HomeLocation(
            location_name="Pippa",
            country="US",
            timezone="America/Los_Angeles",
            latitude=33.86,
            longitude=-118.4,
        )

        self.assertEqual(
            location.web_search_user_location(),
            {
                "type": "approximate",
                "country": "US",
                "timezone": "America/Los_Angeles",
            },
        )

    def test_realtime_context_anchors_here_and_only_uses_rounded_coordinates(self):
        location = HomeLocation(
            location_name="Home",
            country="US",
            timezone="America/Los_Angeles",
            latitude=33.864321,
            longitude=-118.399876,
        )

        instructions = append_home_location_context("Be helpful.", location)

        self.assertTrue(instructions.startswith("Be helpful.\n\n"))
        self.assertIn("'here', 'nearby', 'local'", instructions)
        self.assertIn("33.86, -118.40", instructions)
        self.assertNotIn("33.864321", instructions)
        self.assertNotIn("-118.399876", instructions)

    def test_missing_location_leaves_user_instructions_byte_for_byte_unchanged(self):
        self.assertEqual(
            append_home_location_context("Be helpful.  ", None),
            "Be helpful.  ",
        )

    def test_detects_location_sensitive_search_queries(self):
        self.assertTrue(query_needs_coordinates("What restaurants are near me?"))
        self.assertTrue(query_needs_coordinates("Will it rain tomorrow?"))
        self.assertFalse(query_needs_coordinates("Who won the World Cup in 2018?"))

    def test_derives_direct_and_supervisor_config_urls(self):
        self.assertEqual(
            derive_ha_config_url("http://supervisor/core/api/mcp"),
            "http://supervisor/core/api/config",
        )
        self.assertEqual(
            derive_ha_config_url("https://ha.example:8123/api/mcp/"),
            "https://ha.example:8123/api/config",
        )

    def test_fetch_uses_bearer_auth_and_returns_validated_location(self):
        observed = {}

        def opener(request, timeout):
            observed["authorization"] = request.get_header("Authorization")
            observed["url"] = request.full_url
            observed["timeout"] = timeout
            return _Response(
                b'{"location_name":"Home","country":"US",'
                b'"time_zone":"UTC","latitude":1.2345,"longitude":2.3456}'
            )

        location = fetch_home_location(
            "http://supervisor/core/api/config",
            "secret-token",
            timeout_seconds=1.5,
            opener=opener,
        )

        self.assertIsNotNone(location)
        self.assertEqual(observed["authorization"], "Bearer secret-token")
        self.assertEqual(observed["url"], "http://supervisor/core/api/config")
        self.assertEqual(observed["timeout"], 1.5)

    def test_fetch_failure_is_safe_and_nonfatal(self):
        def opener(request, timeout):
            raise TimeoutError("must-not-propagate")

        with self.assertLogs("app.home_location", logging.WARNING) as captured:
            location = fetch_home_location(
                "http://supervisor/core/api/config",
                "secret-token",
                opener=opener,
            )

        self.assertIsNone(location)
        logs = " ".join(captured.output)
        self.assertIn("TimeoutError", logs)
        self.assertNotIn("secret-token", logs)

    def test_disabled_privacy_gate_does_not_make_an_http_request(self):
        def opener(request, timeout):
            self.fail("disabled location sharing must not call Home Assistant")

        location = load_home_location_if_enabled(
            False,
            "http://supervisor/core/api/config",
            "secret-token",
            opener=opener,
        )

        self.assertIsNone(location)


if __name__ == "__main__":
    unittest.main()
