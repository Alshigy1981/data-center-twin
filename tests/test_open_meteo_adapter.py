"""
Tests for backend/adapters/open_meteo_adapter.py

All HTTP calls are mocked — these tests pass with no internet connection.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from backend.adapters.open_meteo_adapter import OpenMeteoAdapter, TTLCache

# ─────────────────────────────────────────
# Realistic Open-Meteo API response stubs
# ─────────────────────────────────────────

_CURRENT_RESPONSE = {
    "current": {
        "temperature_2m": 14.7,
        "relative_humidity_2m": 62,
    },
    "current_units": {"temperature_2m": "°C", "relative_humidity_2m": "%"},
}

_FORECAST_RESPONSE = {
    "hourly": {
        "time": [f"2024-01-15T{h:02d}:00" for h in range(24)],
        "temperature_2m": [10.0 + h * 0.5 for h in range(24)],
    }
}


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


# ─────────────────────────────────────────
# TTLCache tests
# ─────────────────────────────────────────

class TestTTLCache:
    def test_ttl_cache_valid_within_ttl(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set({"value": 42})
        assert cache.is_valid() is True

    def test_ttl_cache_expired_after_ttl(self):
        cache = TTLCache(ttl_seconds=0)
        cache.set({"value": 42})
        time.sleep(0.01)
        assert cache.is_valid() is False

    def test_ttl_cache_get_returns_none_when_expired(self):
        cache = TTLCache(ttl_seconds=0)
        cache.set({"value": 42})
        time.sleep(0.01)
        assert cache.get() is None

    def test_ttl_cache_stale_returns_value_even_when_expired(self):
        cache = TTLCache(ttl_seconds=0)
        cache.set({"value": 99})
        time.sleep(0.01)
        assert cache.get_stale() == {"value": 99}

    def test_ttl_cache_get_returns_value_within_ttl(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("hello")
        assert cache.get() == "hello"

    def test_ttl_cache_age_increases_over_time(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("x")
        time.sleep(0.05)
        assert cache.age_seconds() >= 0


# ─────────────────────────────────────────
# OpenMeteoAdapter tests
# ─────────────────────────────────────────

class TestOpenMeteoAdapter:

    def _adapter(self, env_overrides: dict | None = None) -> OpenMeteoAdapter:
        """Create adapter with optional env overrides patched in."""
        base_env = {
            "OPEN_METEO_ENABLED":        "true",
            "LOCATION_LAT":              "41.8827",
            "LOCATION_LON":              "-87.6233",
            "LOCATION_NAME":             "Chicago, IL",
            "WEATHER_CACHE_TTL":         "900",
            "CLOUD_API_TIMEOUT_SECONDS": "5",
        }
        if env_overrides:
            base_env.update(env_overrides)
        with patch.dict("os.environ", base_env, clear=False):
            return OpenMeteoAdapter()

    # ── success path ──────────────────────

    def test_get_current_conditions_success(self):
        adapter = self._adapter()
        with patch("httpx.get", return_value=_mock_response(_CURRENT_RESPONSE)):
            result = adapter.get_current_conditions()

        assert result["outside_air_temp_c"] == pytest.approx(14.7)
        assert result["humidity_pct"] == pytest.approx(62.0)
        assert result["source"] == "live"
        assert "timestamp" in result
        assert result["cache_age_seconds"] == 0

    # ── error fallbacks ───────────────────

    def test_get_current_conditions_timeout_returns_fallback(self):
        import httpx as _httpx
        adapter = self._adapter()
        with patch("httpx.get", side_effect=_httpx.TimeoutException("timed out")):
            result = adapter.get_current_conditions()

        assert result["source"] == "simulated"
        assert result["outside_air_temp_c"] == pytest.approx(18.0)
        assert result["humidity_pct"] == pytest.approx(50.0)

    def test_get_current_conditions_http_error_returns_fallback(self):
        import httpx as _httpx
        adapter = self._adapter()
        with patch("httpx.get", side_effect=_httpx.HTTPError("connection refused")):
            result = adapter.get_current_conditions()

        assert result["source"] == "simulated"

    # ── caching ───────────────────────────

    def test_get_current_conditions_uses_cache_on_second_call(self):
        adapter = self._adapter()
        mock_get = MagicMock(return_value=_mock_response(_CURRENT_RESPONSE))
        with patch("httpx.get", mock_get):
            r1 = adapter.get_current_conditions()
            r2 = adapter.get_current_conditions()

        assert mock_get.call_count == 1
        assert r1["outside_air_temp_c"] == r2["outside_air_temp_c"]

    def test_cache_expired_triggers_new_api_call(self):
        adapter = self._adapter({"WEATHER_CACHE_TTL": "0"})
        # TTL=0 means every call is a cache miss after any delay
        mock_get = MagicMock(return_value=_mock_response(_CURRENT_RESPONSE))
        with patch("httpx.get", mock_get):
            adapter.get_current_conditions()
            time.sleep(0.02)
            adapter.get_current_conditions()

        assert mock_get.call_count == 2

    # ── forecast ──────────────────────────

    def test_get_temperature_forecast_24h_returns_24_items(self):
        adapter = self._adapter()
        with patch("httpx.get", return_value=_mock_response(_FORECAST_RESPONSE)):
            result = adapter.get_temperature_forecast_24h()

        assert len(result) == 24
        assert "hour" in result[0]
        assert "temp_c" in result[0]
        assert result[0]["source"] == "live"

    def test_get_temperature_forecast_24h_fallback_on_error(self):
        adapter = self._adapter()
        with patch("httpx.get", side_effect=Exception("network failure")):
            result = adapter.get_temperature_forecast_24h()

        assert len(result) == 24
        assert all(d["source"] == "simulated" for d in result)
        # Verify sinusoidal pattern stays within expected range
        temps = [d["temp_c"] for d in result]
        assert all(10.0 <= t <= 30.0 for t in temps)

    # ── health ────────────────────────────

    def test_is_healthy_false_before_first_call(self):
        adapter = self._adapter()
        assert adapter.is_healthy() is False

    def test_is_healthy_true_after_successful_call(self):
        adapter = self._adapter()
        with patch("httpx.get", return_value=_mock_response(_CURRENT_RESPONSE)):
            adapter.get_current_conditions()
        assert adapter.is_healthy() is True

    def test_get_status_returns_required_keys(self):
        adapter = self._adapter()
        status = adapter.get_status()
        for key in ("enabled", "healthy", "last_fetch", "source", "location", "cache_age_seconds"):
            assert key in status

    # ── disabled mode ─────────────────────

    def test_adapter_disabled_when_env_false(self):
        adapter = self._adapter({"OPEN_METEO_ENABLED": "false"})
        mock_get = MagicMock()
        with patch("httpx.get", mock_get):
            result = adapter.get_current_conditions()

        mock_get.assert_not_called()
        assert result["source"] == "simulated"

    def test_adapter_disabled_forecast_returns_simulated(self):
        adapter = self._adapter({"OPEN_METEO_ENABLED": "false"})
        mock_get = MagicMock()
        with patch("httpx.get", mock_get):
            result = adapter.get_temperature_forecast_24h()

        mock_get.assert_not_called()
        assert len(result) == 24
        assert all(d["source"] == "simulated" for d in result)
