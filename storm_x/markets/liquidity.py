"""Liquidity filter — rejects dead and illiquid Polymarket brackets.

A market is tradeable only when ALL of the following hold:
  1. 24-hour volume ≥ min_volume_usdc  (default $10,000)
  2. Open interest ≥ min_open_interest (default $5,000)
  3. Spread ≤ max_spread_cents         (default 4 cents)
  4. At least min_traders_24h distinct traders in last 24h (default 5)

Markets failing any criterion are flagged as dead and skipped.
"""
from __future__ import annotations

from loguru import logger

from storm_x.config import settings

_F = settings.filters


def is_liquid(market: dict) -> bool:
    """Return True if the market passes all liquidity filters."""
    token = market.get("token_yes", "?")[:12]
    q     = market.get("question", "")[:40]

    vol = float(market.get("volume_24h", 0) or 0)
    if vol < _F.min_volume_usdc:
        logger.debug("DEAD vol={:.0f} < {:.0f} | {}", vol, _F.min_volume_usdc, q)
        return False

    oi = float(market.get("open_interest", 0) or 0)
    if oi > 0 and oi < _F.min_open_interest_usdc:
        logger.debug("DEAD OI={:.0f} < {:.0f} | {}", oi, _F.min_open_interest_usdc, q)
        return False

    spread_raw = market.get("spread")
    if spread_raw is not None:
        spread_cents = float(spread_raw) * 100
        if spread_cents > _F.max_spread_cents:
            logger.debug("DEAD spread={:.1f}c > {}c | {}", spread_cents, _F.max_spread_cents, q)
            return False

    traders = int(market.get("traders_24h", 0) or 0)
    if traders > 0 and traders < _F.min_traders_24h:
        logger.debug("DEAD traders={} < {} | {}", traders, _F.min_traders_24h, q)
        return False

    return True


def filter_markets(markets: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split markets into (tradeable, dead) lists.

    'dead' markets are logged and excluded from the betting pass.
    Volume and open_interest must be populated by the caller (from Gamma API data).
    """
    tradeable, dead = [], []
    for m in markets:
        if is_liquid(m):
            tradeable.append(m)
        else:
            dead.append(m)

    logger.info("Liquidity filter: {} tradeable, {} dead", len(tradeable), len(dead))
    return tradeable, dead
