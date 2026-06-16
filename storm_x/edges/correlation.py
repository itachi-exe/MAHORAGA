"""Cross-city correlation tracker — 60-day rolling Pearson correlation.

When London and Berlin temperature daily-max forecasts are highly correlated
(|r| > 0.80), they are driven by the same synoptic pattern (e.g., a blocking
high over Western Europe).  Betting the same direction in both cities in this
regime reduces effective diversification.

Usage:
  - After each daily cycle, call record_correlation(london_members, berlin_members).
  - Before allocating bets across cities, call effective_n_cities() to get the
    effective independent-bet count (1 when fully correlated, 2 when uncorrelated).
  - The allocator can use this to halve per-city caps when both cities are correlated.

Storage: SQLite table `city_correlation` (city_a, city_b, date, correlation).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from collections import deque

import numpy as np
from loguru import logger

from storm_x.storage.db import _connect as _conn

_WINDOW = 60        # rolling days
_HIGH_CORR = 0.80   # threshold above which cities are treated as correlated

# In-memory deque of (date, correlation) tuples
_history: deque[tuple[date, float]] = deque(maxlen=_WINDOW)
_loaded = False


def _ensure_schema() -> None:
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS city_correlation (
            city_a TEXT NOT NULL,
            city_b TEXT NOT NULL,
            as_of   TEXT NOT NULL,
            corr    REAL NOT NULL,
            PRIMARY KEY (city_a, city_b, as_of)
        )
    """)
    conn.commit()


def _load_history() -> None:
    global _loaded
    if _loaded:
        return
    _ensure_schema()
    conn = _conn()
    rows = conn.execute(
        "SELECT as_of, corr FROM city_correlation "
        "WHERE city_a='london' AND city_b='berlin' "
        "ORDER BY as_of DESC LIMIT ?",
        (_WINDOW,),
    ).fetchall()
    for as_of, corr in reversed(rows):
        _history.append((date.fromisoformat(as_of), corr))
    _loaded = True
    logger.debug("Loaded {} cross-city correlation records", len(_history))


def record_correlation(london_members: np.ndarray, berlin_members: np.ndarray) -> float:
    """Compute and persist correlation between ensemble members for today.

    Members must be aligned (same index = same ensemble run, both cities).
    If lengths differ, use the shorter.
    """
    _ensure_schema()
    n = min(len(london_members), len(berlin_members))
    if n < 2:
        return 0.0

    corr = float(np.corrcoef(london_members[:n], berlin_members[:n])[0, 1])
    if np.isnan(corr):
        corr = 0.0

    today = date.today()
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO city_correlation (city_a, city_b, as_of, corr) VALUES (?,?,?,?)",
        ("london", "berlin", today.isoformat(), round(corr, 4)),
    )
    conn.commit()

    _history.append((today, corr))
    logger.debug("Cross-city correlation today={} r={:.3f}", today.isoformat(), corr)
    return corr


def rolling_mean_correlation() -> float:
    """Return 60-day rolling mean |r| between London and Berlin."""
    _load_history()
    if not _history:
        return 0.0
    return float(np.mean([abs(c) for _, c in _history]))


def effective_n_cities() -> float:
    """Return effective independent city count [1.0, 2.0].

    When |r_mean| ≥ HIGH_CORR, cities share a common driver and count as ~1.
    When uncorrelated, they count as 2 independent bets.
    """
    _load_history()
    r = rolling_mean_correlation()
    # Linear interpolation: 1.0 at r=1.0, 2.0 at r=0.0
    return round(2.0 - r, 3)
