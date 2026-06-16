"""Historical climatology loader — 10 years of daily max temperatures per city.

Data source: Open-Meteo Archive API (free, no key required).
Loaded once into SQLite; subsequent calls read from the DB cache.

Provides: mean and std of daily max temperature for any (city, month, day)
Used by the climatology prior blender to stabilise bracket probabilities at long lead times.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone, timedelta

import httpx
import numpy as np
import pandas as pd
from loguru import logger

from storm_x.config import settings, CityConfig
from storm_x.storage.db import upsert_climatology, get_climatology, climatology_loaded

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_TIMEOUT = httpx.Timeout(30.0)   # archive requests can be slow
_MAX_RETRIES = 3


async def _fetch_archive_chunk(
    client: httpx.AsyncClient,
    cfg: CityConfig,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """Fetch daily temperature_2m_max from Open-Meteo archive for a date range."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = await client.get(_ARCHIVE_URL, params={
                "latitude":    cfg.lat,
                "longitude":   cfg.lon,
                "daily":       "temperature_2m_max",
                "start_date":  start,
                "end_date":    end,
                "timezone":    cfg.timezone,
            })
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == _MAX_RETRIES:
                logger.error("Archive fetch failed {}-{}: {}", start, end, exc)
                return None
            await asyncio.sleep(2 ** attempt)
    return None


async def load_climatology(city: str, force: bool = False) -> bool:
    """Fetch 10 years of daily max temperatures and store mean/std per month-day in SQLite.

    Args:
        city:  City key from config.
        force: Re-fetch even if data already exists.

    Returns True on success, False on failure.
    """
    if not force and climatology_loaded(city):
        logger.info("Climatology already loaded for {} — skipping", city)
        return True

    cfg = settings.city(city)
    years = settings.storage.climatology_years
    today = date.today()
    end_date   = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = today.replace(year=today.year - years).strftime("%Y-%m-%d")

    logger.info("Loading {} years of climatology for {} ({} to {})", years, city, start_date, end_date)

    # Fetch in ~2-year chunks to avoid timeout
    all_dates: list[str] = []
    all_temps: list[float] = []

    chunk_years = 2
    chunk_start = date.fromisoformat(start_date)

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        while chunk_start < date.fromisoformat(end_date):
            chunk_end = min(
                chunk_start.replace(year=chunk_start.year + chunk_years) - timedelta(days=1),
                date.fromisoformat(end_date),
            )
            data = await _fetch_archive_chunk(
                client, cfg,
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
            if data and "daily" in data:
                all_dates.extend(data["daily"]["time"])
                all_temps.extend(
                    float(v) if v is not None else float("nan")
                    for v in data["daily"]["temperature_2m_max"]
                )
                logger.debug("Archive chunk {}-{}: {} rows", chunk_start, chunk_end,
                             len(data["daily"]["time"]))
            chunk_start = chunk_end + timedelta(days=1)

    if not all_dates:
        logger.error("No archive data fetched for {}", city)
        return False

    df = pd.DataFrame({"date": pd.to_datetime(all_dates), "tmax": all_temps})
    df = df.dropna(subset=["tmax"])
    df["month"] = df["date"].dt.month
    df["day"]   = df["date"].dt.day

    grouped = df.groupby(["month", "day"])["tmax"].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(2.0)   # fill NaN std with 2°C prior

    for _, row in grouped.iterrows():
        upsert_climatology(city, int(row["month"]), int(row["day"]),
                           float(row["mean"]), float(row["std"]))

    logger.info("Climatology loaded for {} — {} month-day combos", city, len(grouped))
    return True


def get_climatology_stats(city: str, month: int, day: int) -> tuple[float, float]:
    """Return (mean_tmax, std_tmax) for given city and calendar date.

    Falls back to a flat prior (15°C ± 5°C) if data is unavailable.
    """
    result = get_climatology(city, month, day)
    if result is None:
        logger.warning("No climatology for {} {}-{:02d} — using flat prior", city, month, day)
        return 15.0, 5.0
    return result
