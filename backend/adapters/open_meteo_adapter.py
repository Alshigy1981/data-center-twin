"""
Open-Meteo weather adapter.

Fetches real outside air temperature and humidity from the free, no-key
Open-Meteo API.  Falls back to simulated values silently on any failure.
"""

from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.open-meteo.com/v1/forecast"

# ─────────────────────────────────────────
# TTL cache
# ─────────────────────────────────────────

class TTLCache:
    """Single-value cache with a time-to-live."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._value: Optional[Any] = None
        self._stored_at: Optional[float] = None

    def set(self, value: Any) -> None:
        self._value = value
        self._stored_at = time.monotonic()

    def get(self) -> Optional[Any]:
        """Return cached value if still within TTL, otherwise None."""
        if self._value is None or self._stored_at is None:
            return None
        if self.age_seconds() >= self._ttl:
            return None
        return self._value

    def get_stale(self) -> Optional[Any]:
        """Return cached value regardless of TTL (for graceful fallback)."""
        return self._value

    def is_valid(self) -> bool:
        return self.get() is not None

    def age_seconds(self) -> int:
        if self._stored_at is None:
            return 0
        return int(time.monotonic() - self._stored_at)


# ─────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────

class OpenMeteoAdapter:
    """
    Wraps the Open-Meteo free weather API.

    All public methods are safe to call at any time — they never raise.
    """

    def __init__(self) -> None:
        self._enabled: bool = os.getenv("OPEN_METEO_ENABLED", "true").lower() == "true"
        self._lat: float = float(os.getenv("LOCATION_LAT", "41.8827"))
        self._lon: float = float(os.getenv("LOCATION_LON", "-87.6233"))
        self._location: str = os.getenv("LOCATION_NAME", "Chicago, IL")
        self._ttl: int = int(os.getenv("WEATHER_CACHE_TTL", "900"))
        self._timeout: float = float(os.getenv("CLOUD_API_TIMEOUT_SECONDS", "5"))

        self._cache: TTLCache = TTLCache(self._ttl)
        self._forecast_cache: TTLCache = TTLCache(self._ttl)
        self._last_success: Optional[float] = None

    # ── public API ────────────────────────

    def get_current_conditions(self) -> Dict[str, Any]:
        """
        Return current outside air temperature and humidity.

        Always returns a valid dict — falls back to simulated on any error.
        """
        cached = self._cache.get()
        if cached is not None:
            result = dict(cached)
            result["cache_age_seconds"] = self._cache.age_seconds()
            return result

        if not self._enabled:
            return self._simulated_conditions()

        try:
            data = self._fetch_current()
            result = {
                "outside_air_temp_c": float(data["current"]["temperature_2m"]),
                "humidity_pct":       float(data["current"]["relative_humidity_2m"]),
                "location":           self._location,
                "timestamp":          datetime.now(timezone.utc).isoformat(),
                "source":             "live",
                "cache_age_seconds":  0,
            }
            self._cache.set(result)
            self._last_success = time.monotonic()
            logger.info(
                "Open-Meteo: %.1f°C / %.0f%% RH at %s",
                result["outside_air_temp_c"],
                result["humidity_pct"],
                self._location,
            )
            return result

        except Exception as exc:
            logger.warning("Open-Meteo current fetch failed: %s", exc)
            stale = self._cache.get_stale()
            if stale is not None:
                result = dict(stale)
                result["source"] = "simulated"
                result["cache_age_seconds"] = self._cache.age_seconds()
                return result
            return self._simulated_conditions()

    def get_temperature_forecast_24h(self) -> List[Dict[str, Any]]:
        """
        Return next 24 hours of hourly temperature forecasts.

        Falls back to a sinusoidal 15–25°C pattern on any error.
        """
        cached = self._forecast_cache.get()
        if cached is not None:
            return cached

        if not self._enabled:
            return self._simulated_forecast()

        try:
            data = self._fetch_forecast()
            hourly = data.get("hourly", {})
            times  = hourly.get("time", [])
            temps  = hourly.get("temperature_2m", [])
            result = [
                {"hour": t, "temp_c": float(v), "source": "live"}
                for t, v in zip(times, temps)
            ][:24]
            if len(result) < 24:
                result = self._simulated_forecast()
            self._forecast_cache.set(result)
            return result

        except Exception as exc:
            logger.warning("Open-Meteo forecast fetch failed: %s", exc)
            stale = self._forecast_cache.get_stale()
            if stale is not None:
                return stale
            return self._simulated_forecast()

    def is_healthy(self) -> bool:
        """True if the last successful API call was within the past 30 minutes."""
        if self._last_success is None:
            return False
        return (time.monotonic() - self._last_success) < 1800

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled":            self._enabled,
            "healthy":            self.is_healthy(),
            "last_fetch":         (
                datetime.fromtimestamp(
                    time.time() - (time.monotonic() - self._last_success),
                    tz=timezone.utc,
                ).isoformat()
                if self._last_success is not None else None
            ),
            "source":             "live" if self.is_healthy() else "simulated",
            "location":           self._location,
            "cache_age_seconds":  self._cache.age_seconds(),
        }

    # ── private helpers ───────────────────

    def _fetch_current(self) -> Dict[str, Any]:
        params = {
            "latitude":     self._lat,
            "longitude":    self._lon,
            "current":      "temperature_2m,relative_humidity_2m",
            "forecast_days": 1,
            "timezone":     "auto",
        }
        resp = httpx.get(_ENDPOINT, params=params, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def _fetch_forecast(self) -> Dict[str, Any]:
        params = {
            "latitude":      self._lat,
            "longitude":     self._lon,
            "hourly":        "temperature_2m",
            "forecast_days": 2,
            "timezone":      "auto",
        }
        resp = httpx.get(_ENDPOINT, params=params, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def _simulated_conditions(self) -> Dict[str, Any]:
        return {
            "outside_air_temp_c": 18.0,
            "humidity_pct":       50.0,
            "location":           self._location,
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "source":             "simulated",
            "cache_age_seconds":  0,
        }

    @staticmethod
    def _simulated_forecast() -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        result = []
        for h in range(24):
            temp = 20.0 + 5.0 * math.sin(math.pi * (h - 6) / 12)
            ts = now.replace(hour=(now.hour + h) % 24, minute=0, second=0, microsecond=0)
            result.append({"hour": ts.isoformat(), "temp_c": round(temp, 1), "source": "simulated"})
        return result
