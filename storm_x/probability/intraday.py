"""Intraday Bayesian updater — conditions ensemble on observed running daily max.

After 10 AM local time at the resolution station, the running maximum temperature
observed so far today constrains the possible final outcomes.

Key insight:
  - If running_max = 22.4°C at 2 PM, the daily max CANNOT end below 22.4°C.
    Any ensemble member predicting tmax < 22.4°C is eliminated.
  - If a bracket requires tmax < 20°C and running_max = 22°C, that bracket
    has P = 0 and we skip betting on it entirely.
  - We restrict the ensemble to members where member_tmax ≥ running_max,
    then recompute bracket probabilities from this conditioned subset.

Called every 30 minutes after 10 AM local station time.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from loguru import logger

from storm_x.config import settings
from storm_x.probability.brackets import Bracket, bracket_probability


def should_update(city: str) -> bool:
    """Return True if it is 10 AM or later in local station time."""
    cfg = settings.city(city)
    tz  = ZoneInfo(cfg.timezone)
    return datetime.now(tz).hour >= 10


def condition_members(members: np.ndarray, running_max: float) -> np.ndarray:
    """Remove ensemble members inconsistent with the observed running max.

    Any member predicting a daily max below the already-observed running max
    is physically impossible and is discarded.

    Args:
        members:     Bias-corrected ensemble member array.
        running_max: Highest temperature recorded at station so far today (°C).

    Returns:
        Subset of members where member >= running_max.  If this subset is
        empty (all members below running_max — rare model failure), the
        original array is returned with a warning.
    """
    conditioned = members[members >= running_max]
    if len(conditioned) == 0:
        logger.warning(
            "Intraday: all {} members below running_max={:.1f}°C — keeping original",
            len(members), running_max,
        )
        return members
    logger.debug(
        "Intraday: conditioned {} → {} members on running_max={:.1f}°C",
        len(members), len(conditioned), running_max,
    )
    return conditioned


def is_bracket_eliminated(bracket: Bracket, running_max: float) -> bool:
    """Return True if the bracket is already impossible given the running max.

    A bracket is impossible if its entire range is below running_max.
    """
    return bracket.hi <= running_max


def updated_probability(
    members: np.ndarray,
    bracket: Bracket,
    running_max: float,
    n_brackets: int,
    city: str,
    target_month: int,
    target_day: int,
    lead_hours: float,
) -> float:
    """Recompute bracket probability conditioned on running_max.

    Returns 0.0 immediately if the bracket is already eliminated.
    """
    if is_bracket_eliminated(bracket, running_max):
        logger.debug("Bracket {} eliminated: running_max={:.1f} ≥ bracket.hi={:.1f}",
                     bracket.label(), running_max, bracket.hi)
        return 0.0

    conditioned = condition_members(members, running_max)
    return bracket_probability(
        conditioned, bracket, n_brackets, city, target_month, target_day, lead_hours
    )
