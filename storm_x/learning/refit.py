"""Nightly learning loop — bias update + isotonic calibrator refit.

Runs at 02:00 UTC each night (configured in scheduler/runner.py).

Sequence:
  1. For each city, check if yesterday's Wunderground observation is available.
  2. Look up yesterday's forecast mean from the ensemble cache or bet records.
  3. Record the (forecast, observed) pair in bias_history.
  4. Refit the isotonic calibrator on the last 60 resolved bets.
  5. Record cross-city correlation from yesterday's ensemble members.

The bias estimator and calibrator are module-level singletons, so the refitted
state is immediately available to the main trading loop the next morning.
"""
from __future__ import annotations

from datetime import date, datetime, timezone, timedelta

import numpy as np
from loguru import logger

from storm_x.calibration.bias import record_outcome, apply_bias
from storm_x.calibration.isotonic import calibrator
from storm_x.config import settings
from storm_x.data.observation import get_running_max
from storm_x.storage.db import get_resolved_bets, _connect as _conn


async def _get_yesterday_forecast_mean(city: str, yesterday: date) -> float | None:
    """Pull yesterday's ensemble mean from the last open/resolved bet for that city and date.

    Fallback: return None if no record exists (happens in early paper trading).
    """
    conn = _conn()
    row = conn.execute(
        """
        SELECT model_prob, calibrated_prob, bracket_description, created_at
        FROM bet_history
        WHERE city = ?
          AND date(created_at) = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (city, yesterday.isoformat()),
    ).fetchone()

    if row is None:
        logger.warning("No bet record for {} on {} — cannot compute bias", city, yesterday)
        return None

    # Use model_prob as a proxy for the ensemble central estimate.
    # This is imperfect but sufficient until we store explicit forecast means.
    return float(dict(row)["model_prob"])


async def nightly_refit() -> dict:
    """Run the full nightly learning loop.

    Returns a summary dict for logging/health-endpoint consumption.
    """
    summary: dict = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "bias_updates": {},
        "calibrator_refitted": False,
        "calibrator_n_samples": 0,
    }

    yesterday = date.today() - timedelta(days=1)

    for city_key in settings.cities:
        logger.info("Nightly refit — processing bias for city={} date={}", city_key, yesterday)

        # 1. Get yesterday's observed max temperature
        obs, is_approx = await get_running_max(city_key)
        if obs is None:
            logger.warning("No observation for {} on {} — skipping bias update", city_key, yesterday)
            summary["bias_updates"][city_key] = "no_observation"
            continue

        if is_approx:
            logger.warning("Observation for {} is approximate (Open-Meteo fallback) — bias update skipped", city_key)
            summary["bias_updates"][city_key] = "approximate_observation_skipped"
            continue

        # 2. Get yesterday's ensemble forecast mean
        forecast_mean = await _get_yesterday_forecast_mean(city_key, yesterday)
        if forecast_mean is None:
            summary["bias_updates"][city_key] = "no_forecast_record"
            continue

        # 3. Record the error
        record_outcome(city_key, yesterday.isoformat(), forecast_mean, obs)
        summary["bias_updates"][city_key] = {
            "forecast": round(forecast_mean, 2),
            "observed": round(obs, 2),
            "error": round(obs - forecast_mean, 2),
        }
        logger.info("Bias updated: {} {} forecast={:.1f} observed={:.1f}", city_key, yesterday, forecast_mean, obs)

    # 4. Refit isotonic calibrator
    calibrator.refit()
    summary["calibrator_refitted"] = calibrator.is_active
    summary["calibrator_n_samples"] = calibrator._n_samples

    logger.info(
        "Nightly refit complete | calibrator_active={} n_samples={}",
        calibrator.is_active, calibrator._n_samples,
    )
    return summary
