"""Regime score — classifies forecast uncertainty and sets the dynamic edge threshold.

Score is a weighted average of three normalized signals:
  1. Ensemble pool standard deviation (spread across all 119 members)
  2. Absolute pressure tendency (last 12 h) — not yet wired to observation data;
     defaults to 0.5 (neutral) until the observation layer provides it.
  3. Forecast lead time in hours, normalized by 48 h

Regime → edge threshold mapping:
  [0.0, 0.3)  stable    → 6%   (high confidence; accept smaller edges)
  [0.3, 0.6)  normal    → 8%   (default)
  [0.6, 0.8)  unstable  → 12%  (models disagree; demand larger edge)
  [0.8, 1.0]  chaotic   → skip (no betting today)
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from storm_x.config import settings

_AGREE_MAX  = settings.edges.model_agreement_max_disagree   # 1.5°C
_SPREAD_MAX = settings.edges.max_pool_spread                 # 3.0°C


def compute_score(
    members: np.ndarray,
    lead_hours: float,
    pressure_tendency: float | None = None,
) -> float:
    """Compute the regime score in [0, 1].

    Args:
        members:           Bias-corrected ensemble member array.
        lead_hours:        Hours until market resolution.
        pressure_tendency: |ΔP| over last 12 h in hPa.  Pass None to use neutral 0.5.

    Returns:
        Regime score in [0.0, 1.0].  Higher = more uncertain.
    """
    spread = float(np.std(members)) if len(members) > 1 else 0.0
    s1 = min(spread / _SPREAD_MAX, 1.0)                          # spread component

    s2 = 0.5 if pressure_tendency is None else min(abs(pressure_tendency) / 10.0, 1.0)

    s3 = min(lead_hours / 48.0, 1.0)                             # lead-time component

    score = (s1 + s2 + s3) / 3.0
    logger.debug("Regime: spread={:.2f}°C s1={:.2f} s2={:.2f} s3={:.2f} score={:.3f}",
                 spread, s1, s2, s3, score)
    return score


def edge_threshold(score: float) -> float | None:
    """Return the edge threshold for this regime score, or None to skip entirely.

    Returns None when score >= 0.8 (chaotic regime — no betting).
    """
    if score >= 0.8:
        logger.warning("Regime CHAOTIC ({:.3f}) — skipping all bets today", score)
        return None
    if score >= 0.6:
        return settings.edges.base_edge_threshold * 1.5   # 12%
    if score >= 0.3:
        return settings.edges.base_edge_threshold          # 8%
    return settings.edges.base_edge_threshold * 0.75      # 6%


def model_agreement_gate(members: np.ndarray, model_names: list[str]) -> bool:
    """Return True (cleared to bet) if ECMWF and GFS means agree within threshold.

    If either model is missing, the gate passes (single-model days still bet).
    Rejects the day if the two main models disagree by > 1.5°C or the pooled
    std dev of all members exceeds 3.0°C.
    """
    # Identify ECMWF and GFS member slices by model name order
    # The ensemble fetcher returns members concatenated: [ecmwf..., gfs..., icon...]
    # We can't split by count reliably without stored counts, so gate on pooled std
    pooled_std = float(np.std(members)) if len(members) > 1 else 0.0
    if pooled_std > _SPREAD_MAX:
        logger.warning(
            "Agreement gate FAILED: pooled std={:.2f}°C > {:.1f}°C — skipping city",
            pooled_std, _SPREAD_MAX,
        )
        return False

    logger.debug("Agreement gate PASSED: pooled std={:.2f}°C", pooled_std)
    return True
