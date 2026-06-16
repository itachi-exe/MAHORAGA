"""Empirical bracket probability — the central probability function of STORM-X.

Pipeline for a single bracket:
  1. Count ensemble members landing in the bracket (after bias correction)
  2. Apply Laplace smoothing to prevent 0/1 extremes
  3. Blend with climatology prior weighted by lead time
  4. Apply isotonic calibration (once ≥60 resolved bets exist)

Bracket definitions (all temperatures in °C, matching Polymarket resolution):
  - Exact N:           [N.0, N+1.0)   "highest temp is N°C"
  - Lower bound N:     (-∞, N+1.0)    "N°C or below"  → below N+1
  - Upper bound N:     [N.0, +∞)      "N°C or above"  → at least N

The spec says:
  exact 19 → "at least 19.0 and below 20.0"   → [19, 20)
  lower 19 → "below 20.0"                       → (-∞, 20)
  upper 21 → "at least 21.0"                    → [21, ∞)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from storm_x.calibration.smoothing import laplace, clamp
from storm_x.calibration.isotonic import calibrator
from storm_x.calibration.prior import blend, climatology_bracket_prob
from storm_x.data.climatology import get_climatology_stats


@dataclass(frozen=True)
class Bracket:
    temp_value: float
    is_lower_bound: bool   # "N°C or below"
    is_upper_bound: bool   # "N°C or above"

    @property
    def lo(self) -> float:
        if self.is_lower_bound:
            return float("-inf")
        return self.temp_value          # exact or upper: starts at N

    @property
    def hi(self) -> float:
        if self.is_upper_bound:
            return float("inf")
        return self.temp_value + 1.0    # exact or lower: ends at N+1

    def contains(self, temp: float) -> bool:
        return self.lo <= temp < self.hi

    def label(self) -> str:
        if self.is_lower_bound:
            return f"≤{self.temp_value:.0f}°C"
        if self.is_upper_bound:
            return f"≥{self.temp_value:.0f}°C"
        return f"={self.temp_value:.0f}°C"


def bracket_probability(
    members: np.ndarray,
    bracket: Bracket,
    n_brackets: int,
    city: str,
    target_month: int,
    target_day: int,
    lead_hours: float,
) -> float:
    """Compute the full calibrated probability that tomorrow's tmax lands in bracket.

    Args:
        members:       Bias-corrected ensemble member array, shape (N,).
        bracket:       The bracket definition.
        n_brackets:    Total number of brackets for this market (Laplace denominator).
        city:          City key (for climatology lookup).
        target_month:  Calendar month of forecast date.
        target_day:    Calendar day of forecast date.
        lead_hours:    Hours until market resolution.

    Returns:
        Calibrated probability in (0.001, 0.999).
    """
    if len(members) == 0:
        return 1.0 / max(n_brackets, 1)

    # Step 1 — empirical count
    count = int(np.sum([bracket.contains(float(m)) for m in members]))
    total = len(members)

    # Step 2 — Laplace smoothing
    p_raw = laplace(count, total, n_brackets)

    # Step 3 — climatology blend
    clim_mean, clim_std = get_climatology_stats(city, target_month, target_day)
    p_clim = climatology_bracket_prob(clim_mean, clim_std, bracket.lo, bracket.hi)
    p_blended = blend(p_raw, p_clim, lead_hours)

    # Step 4 — isotonic calibration
    p_cal = calibrator.calibrate(p_blended)
    p_final = clamp(p_cal)

    logger.debug(
        "P({}) count={}/{} raw={:.3f} clim={:.3f} blended={:.3f} cal={:.3f}",
        bracket.label(), count, total, p_raw, p_clim, p_blended, p_final,
    )
    return p_final


def bracket_from_market(market: dict) -> Bracket:
    """Build a Bracket object from a market dict (as returned by markets.discovery)."""
    return Bracket(
        temp_value=float(market["temp_value"]),
        is_lower_bound=bool(market["is_lower_bound"]),
        is_upper_bound=bool(market["is_upper_bound"]),
    )
