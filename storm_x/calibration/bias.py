"""Rolling additive bias estimator — corrects ensemble mean for station vs grid-point offset.

Bias is computed per city per calendar month from a 90-day rolling window of
(forecast − observed) errors stored in SQLite.  Applied as a simple subtraction
to every ensemble member before any probability computation.

Example: if ECMWF/GFS/ICON typically over-predict London tmax by +2.1°C in June,
all members are shifted down by 2.1°C so bracket probability counts are centred
on what EGLC will actually record.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from loguru import logger

from storm_x.config import settings
from storm_x.storage.db import get_bias_rows, insert_bias_row


def get_current_bias(city: str, month: int) -> float:
    """Return the rolling mean forecast error for (city, month).

    Positive bias → ensemble over-predicts → subtract from members.
    Returns 0.0 if fewer than 5 samples exist (no correction applied).
    """
    rows = get_bias_rows(city, month, window_days=settings.bias.rolling_window_days)
    if not rows:
        return 0.0

    # Filter to current + adjacent months within rolling window
    cutoff = (datetime.now(timezone.utc) - timedelta(days=settings.bias.rolling_window_days)).date()
    errors = [
        r["error"] for r in rows
        if r["forecast_date"] >= str(cutoff)
    ]

    if len(errors) < 5:
        logger.debug("Bias {} month={}: only {} samples — returning 0.0", city, month, len(errors))
        return 0.0

    bias = sum(errors) / len(errors)
    logger.debug("Bias {} month={}: {:.2f}°C from {} samples", city, month, bias, len(errors))
    return bias


def apply_bias(members: "np.ndarray", city: str, target_date: datetime) -> "np.ndarray":
    """Subtract the rolling additive bias from all ensemble members.

    This corrects for the systematic offset between the Open-Meteo grid point
    and the Wunderground resolution station (e.g. EGLC urban heat island).

    Args:
        members:     Raw ensemble daily-max array, shape (N,).
        city:        City key.
        target_date: The date being forecast (used to determine calendar month).

    Returns:
        Bias-corrected member array, same shape.
    """
    import numpy as np
    month = target_date.month
    bias  = get_current_bias(city, month)
    if bias != 0.0:
        logger.info("Applying bias correction: {:.2f}°C for {} month={}", bias, city, month)
    return members - bias


def record_outcome(city: str, forecast_date: str, forecast_val: float, observed_val: float) -> None:
    """Append a resolved forecast error to the bias history table.

    Called by the nightly refit after each day's Wunderground observation is final.
    """
    month = datetime.fromisoformat(forecast_date).month
    insert_bias_row(city, month, forecast_date, forecast_val, observed_val)
    logger.info("Bias record: {} {} forecast={:.1f} observed={:.1f} error={:+.2f}",
                city, forecast_date, forecast_val, observed_val, observed_val - forecast_val)
