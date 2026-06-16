"""Tail bucket detector — identifies markets priced below 15c as potential value plays.

A market priced <15c implies the crowd gives <15% probability to that outcome.
If our ensemble gives ≥21% (i.e., 6% edge at 15c price), the implied odds are
mispriced.  Tail brackets also tend to have the widest bid-ask spreads, so we
apply the tighter edge threshold (6%) to partially compensate.

Key heuristic:
  tail_bucket(market) = True  iff  market_price < TAIL_THRESHOLD (0.15)
  eligible_threshold  = TAIL_EDGE_THRESHOLD (0.06)  instead of standard threshold

This module does NOT modify edge computation — it only classifies whether a
market qualifies as a tail bucket so the caller can pass the lower threshold.
"""
from __future__ import annotations

from storm_x.config import settings

_TAIL_THRESHOLD      = settings.edges.tail_threshold_price    # 0.15
_TAIL_EDGE_THRESHOLD = settings.edges.tail_edge_threshold      # 0.06
_STANDARD_THRESHOLD  = settings.edges.base_edge_threshold      # 0.08


def is_tail_bucket(market_price: float) -> bool:
    """Return True if this market is priced below the tail threshold."""
    return market_price < _TAIL_THRESHOLD


def effective_edge_threshold(market_price: float, regime_edge_threshold: float | None) -> float | None:
    """Return the edge threshold to apply for this market price.

    Args:
        market_price:           Current market price for the outcome.
        regime_edge_threshold:  Threshold returned by regime scorer (None = chaotic, do not bet).

    Returns:
        Effective threshold or None if betting is disabled.
    """
    if regime_edge_threshold is None:
        return None

    if is_tail_bucket(market_price):
        # Use tail threshold but never looser than what the regime allows
        return min(_TAIL_EDGE_THRESHOLD, regime_edge_threshold)

    return regime_edge_threshold
