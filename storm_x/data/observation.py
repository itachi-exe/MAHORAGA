"""Live observation puller — running daily maximum temperature at the resolution station.

Primary:  Wunderground HTML scrape for EGLC (London City Airport).
Fallback: Open-Meteo current forecast at station coordinates.

The running max is the highest temperature recorded at the station so far
today in local time.  After ~10 AM local time the intraday Bayesian updater
uses this to condition ensemble probabilities.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
from loguru import logger

from storm_x.config import settings, CityConfig

_TIMEOUT = httpx.Timeout(10.0)
_MAX_RETRIES = 3

# Wunderground observation history URL template
_WU_HISTORY_URL = "https://www.wunderground.com/history/daily/{path}"
# Open-Meteo current conditions fallback
_OM_CURRENT_URL = "https://api.open-meteo.com/v1/forecast"


_F_KEYS = ("temperatureMaxSince7Am", "temperatureMax24Hour", "temperature")


def _f_to_c(f: float) -> float:
    return round((f - 32) * 5 / 9, 1)


async def _wu_running_max(client: httpx.AsyncClient, cfg: CityConfig) -> float | None:
    """Scrape Wunderground daily history page for today's running max temperature (°C).

    Wunderground embeds a JSON blob in a <script type="application/json"> tag.
    The key 'temperatureMaxSince7Am' holds the running daily max in Fahrenheit.
    Falls through to None if the page structure changes or is unavailable.
    """
    import json as _json
    url = _WU_HISTORY_URL.format(path=cfg.wunderground_path)
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; STORMX/1.0)"},
                follow_redirects=True,
            )
            resp.raise_for_status()
            html = resp.text
            break
        except Exception as exc:
            if attempt == _MAX_RETRIES:
                logger.warning("Wunderground scrape failed after {} attempts: {}", _MAX_RETRIES, exc)
                return None
            await asyncio.sleep(2 ** attempt)

    # Parse the embedded JSON blob (Wunderground SPA data store)
    json_blocks = re.findall(r'<script[^>]+type=["\']application/json["\'][^>]*>(.*?)</script>',
                             html, re.DOTALL)
    raw_json = " ".join(json_blocks)

    for key in _F_KEYS:
        m = re.search(rf'"{key}"\s*:\s*(-?\d+\.?\d*)', raw_json)
        if m:
            val_f = float(m.group(1))
            # Sanity check: Fahrenheit range plausible for London/Berlin: -4°F to 113°F
            if -4 <= val_f <= 113:
                val_c = _f_to_c(val_f)
                logger.info("Wunderground {} | {} = {:.1f}°F → {:.1f}°C",
                            cfg.station, key, val_f, val_c)
                return val_c

    logger.warning("Wunderground {} | could not parse temperature from JSON blob", cfg.station)
    return None


async def _om_running_max(client: httpx.AsyncClient, cfg: CityConfig) -> float | None:
    """Fallback: Open-Meteo hourly temperature for today; return max of past hours."""
    tz = ZoneInfo(cfg.timezone)
    local_now = datetime.now(tz)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = await client.get(_OM_CURRENT_URL, params={
                "latitude":   cfg.lat,
                "longitude":  cfg.lon,
                "hourly":     "temperature_2m",
                "past_days":  0,
                "forecast_days": 1,
                "timezone":   cfg.timezone,
            })
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            if attempt == _MAX_RETRIES:
                logger.error("OM fallback failed: {}", exc)
                return None
            await asyncio.sleep(2 ** attempt)

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    temps  = hourly.get("temperature_2m", [])

    today_str   = local_now.strftime("%Y-%m-%d")
    current_hour = local_now.hour

    past_temps = [
        t for time_str, t in zip(times, temps)
        if time_str.startswith(today_str)
        and int(time_str[11:13]) <= current_hour
        and t is not None
    ]

    if not past_temps:
        return None

    running_max = max(past_temps)
    logger.info("OM fallback {} | running max {:.1f}°C (APPROXIMATE)", cfg.station, running_max)
    return running_max


async def get_running_max(city: str) -> tuple[float | None, bool]:
    """Return (running_daily_max, is_approximate) for the given city's resolution station.

    Tries Wunderground first; if it fails, falls back to Open-Meteo.
    'is_approximate' is True when falling back to Open-Meteo.
    """
    cfg = settings.city(city)

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        wu_result = await _wu_running_max(client, cfg)
        if wu_result is not None:
            return wu_result, False

        logger.warning("{} | Wunderground failed — using Open-Meteo fallback (APPROXIMATE)", city)
        om_result = await _om_running_max(client, cfg)
        return om_result, True
