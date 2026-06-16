"""Fetch 120+ ensemble member temperature forecasts from Open-Meteo.

Three models are queried per city per cycle:
  ecmwf_ifs025  → 50 members   (ECMWF IFS 0.25° ensemble)
  gfs_seamless  → 30 members   (GFS GEFS)
  icon_seamless → 39 members   (ICON EPS)
  Total         → ~119 members

Hourly temperature_2m is requested for forecast_days=2.  The daily maximum
for each member over the TARGET day (tomorrow UTC) is extracted and returned
as a single concatenated numpy array.

Results are cached per (city, hour, model) in SQLite with a 10-minute TTL.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import NamedTuple

import httpx
import numpy as np
from loguru import logger

from storm_x.config import settings, CityConfig
from storm_x.storage.db import cache_get_ensemble, cache_set_ensemble, _connect

_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
_TIMEOUT = httpx.Timeout(10.0)
_MAX_RETRIES = 3


class EnsembleResult(NamedTuple):
    members: np.ndarray           # shape (N,) — daily max per member, bias-corrected AFTER this layer
    member_count: int
    model_names: list[str]
    generation_hash: str          # SHA1 of raw member array — changes when new model run drops
    fetched_at: datetime


async def _fetch_model_members(
    client: httpx.AsyncClient,
    city_cfg: CityConfig,
    model_name: str,
    target_date: datetime,
) -> list[float]:
    """Fetch hourly temperature_2m for all members of one model, return daily-max per member."""
    hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
    cached = cache_get_ensemble(city_cfg.station, hour_key, model_name)
    if cached is not None:
        logger.debug("Cache hit: {} {} {}", city_cfg.station, model_name, hour_key)
        return cached

    params = {
        "latitude":     city_cfg.lat,
        "longitude":    city_cfg.lon,
        "hourly":       "temperature_2m",
        "models":       model_name,
        "forecast_days": 2,
        "timezone":     "UTC",
    }

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = await client.get(_ENSEMBLE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            if attempt == _MAX_RETRIES:
                logger.error("Ensemble fetch failed {} {} after {} attempts: {}",
                             city_cfg.station, model_name, _MAX_RETRIES, exc)
                return []
            wait = 2 ** attempt
            logger.warning("Ensemble fetch attempt {}/{} failed: {} — retry in {}s",
                           attempt, _MAX_RETRIES, exc, wait)
            await asyncio.sleep(wait)

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        logger.warning("No hourly data for {} {}", city_cfg.station, model_name)
        return []

    target_str = target_date.strftime("%Y-%m-%d")
    member_keys = sorted(k for k in hourly if k.startswith("temperature_2m_member"))

    if not member_keys:
        logger.warning("No member columns for {} {} — skipping", city_cfg.station, model_name)
        return []

    members_daily_max: list[float] = []
    for mk in member_keys:
        vals = hourly[mk]
        daily_vals = [
            v for t, v in zip(times, vals)
            if t.startswith(target_str) and v is not None
        ]
        if daily_vals:
            members_daily_max.append(float(np.max(daily_vals)))

    logger.info("Model {} | {} | {} members | target={}",
                model_name, city_cfg.station, len(members_daily_max), target_str)

    cache_set_ensemble(city_cfg.station, hour_key, model_name, members_daily_max)
    return members_daily_max


async def fetch_ensemble(city: str, target_date: datetime | None = None) -> EnsembleResult:
    """Fetch and concatenate ensemble members from all configured models for one city.

    Args:
        city: City key from config ('london' or 'berlin').
        target_date: The date whose daily max we're forecasting.  Defaults to tomorrow UTC.

    Returns:
        EnsembleResult with concatenated member array and metadata.
    """
    cfg = settings.city(city)
    if target_date is None:
        target_date = datetime.now(timezone.utc) + timedelta(days=1)

    all_members: list[float] = []
    model_names: list[str]   = []

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        tasks = [
            _fetch_model_members(client, cfg, m.name, target_date)
            for m in settings.ensemble.models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for model_cfg, result in zip(settings.ensemble.models, results):
        if isinstance(result, Exception):
            logger.error("Model {} raised exception: {}", model_cfg.name, result)
            continue
        if result:
            all_members.extend(result)
            model_names.append(model_cfg.name)

    arr = np.array(all_members, dtype=np.float32)
    gen_hash = hashlib.sha1(arr.tobytes()).hexdigest()[:16]

    logger.info(
        "Ensemble total: {} members from {} models | city={} | hash={}",
        len(arr), len(model_names), city, gen_hash,
    )

    _snapshot_to_db(city, arr, model_names, gen_hash)

    return EnsembleResult(
        members=arr,
        member_count=len(arr),
        model_names=model_names,
        generation_hash=gen_hash,
        fetched_at=datetime.now(timezone.utc),
    )


def _snapshot_to_db(city: str, members: np.ndarray, model_names: list[str], gen_hash: str) -> None:
    """Persist ensemble snapshot to market_state table for change detection."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO market_state (city, fetch_time, members_json, member_count, model_names, generation_hash) "
            "VALUES (?,?,?,?,?,?)",
            (
                city,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(members.tolist()),
                len(members),
                json.dumps(model_names),
                gen_hash,
            ),
        )


def last_generation_hash(city: str) -> str | None:
    """Return the most recent generation_hash for this city, or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT generation_hash FROM market_state WHERE city=? ORDER BY fetch_time DESC LIMIT 1",
            (city,),
        ).fetchone()
    return row["generation_hash"] if row else None
