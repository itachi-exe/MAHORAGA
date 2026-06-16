"""Polymarket live price fetcher — uses CLOB get_price endpoint exclusively.

get_order_book is NOT used: it returns stale ghost prices that can be minutes
behind actual market state.  get_price with side=BUY reflects the true ask.

Prices are cached for 30 seconds per token_id.
"""
from __future__ import annotations

import time
from functools import lru_cache

import httpx
from loguru import logger

_CLOB_BASE   = "https://clob.polymarket.com"
_TIMEOUT     = httpx.Timeout(10.0)
_MAX_RETRIES = 3
_CACHE_TTL   = 30.0   # seconds

# Simple TTL cache: {token_id: (price, timestamp)}
_price_cache: dict[str, tuple[float, float]] = {}


async def get_price(token_id: str, side: str = "BUY") -> float | None:
    """Fetch current price for a token_id from the CLOB.

    Args:
        token_id: YES or NO token ID from Polymarket.
        side:     'BUY' (returns ask) or 'SELL' (returns bid).

    Returns:
        Price as a float in [0.001, 0.999], or None on failure.
    """
    cache_key = f"{token_id}:{side}"
    cached = _price_cache.get(cache_key)
    if cached and (time.time() - cached[1]) < _CACHE_TTL:
        return cached[0]

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await client.get(
                    f"{_CLOB_BASE}/price",
                    params={"token_id": token_id, "side": side},
                )
                resp.raise_for_status()
                data = resp.json()
                price_str = data.get("price") or data.get("Price")
                if price_str is None:
                    logger.warning("No price field in response for {}", token_id[:20])
                    return None
                price = float(price_str)
                price = max(0.001, min(0.999, price))
                _price_cache[cache_key] = (price, time.time())
                return price
            except Exception as exc:
                if attempt == _MAX_RETRIES:
                    logger.error("get_price failed {} after {} attempts: {}",
                                 token_id[:20], _MAX_RETRIES, exc)
                    return None
                import asyncio
                await asyncio.sleep(2 ** attempt)
    return None


async def get_spread(token_yes: str) -> float | None:
    """Return bid-ask spread in cents for a YES token.

    Spread = ask_price - bid_price.  Returned as a float (e.g. 0.03 = 3 cents).
    Returns None if either side fails.
    """
    import asyncio
    ask, bid = await asyncio.gather(
        get_price(token_yes, "BUY"),
        get_price(token_yes, "SELL"),
    )
    if ask is None or bid is None:
        return None
    return round(ask - bid, 4)


async def enrich_market_prices(market: dict) -> dict:
    """Update a market dict with live YES/NO prices and spread."""
    import asyncio
    yes_ask, spread = await asyncio.gather(
        get_price(market["token_yes"], "BUY"),
        get_spread(market["token_yes"]),
    )
    if yes_ask is not None:
        market = {**market, "yes_ask": yes_ask, "yes_price": yes_ask}
    if spread is not None:
        market = {**market, "spread": spread}
    return market
