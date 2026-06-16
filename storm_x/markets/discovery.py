"""Polymarket market discovery via the Gamma API event slug.

Slug pattern (identical to legacy STORM, predictable daily):
  highest-temperature-in-london-on-june-16-2026
  highest-temperature-in-berlin-on-june-16-2026

Returns one list of market dicts per city per day, enriched with parsed
bracket definitions (temp_value, is_lower_bound, is_upper_bound).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta

import httpx
from loguru import logger

from storm_x.config import settings

_GAMMA_BASE  = "https://gamma-api.polymarket.com"
_TIMEOUT     = httpx.Timeout(10.0)
_MAX_RETRIES = 3
_CITY_SLUGS  = {
    "london": "highest-temperature-in-london-on-{month}-{day}-{year}",
    "berlin": "highest-temperature-in-berlin-on-{month}-{day}-{year}",
}


def _build_slug(city: str, date: datetime) -> str:
    month = date.strftime("%B").lower()
    day   = str(date.day)
    year  = str(date.year)
    pattern = _CITY_SLUGS.get(city, "")
    return pattern.format(month=month, day=day, year=year)


def _parse_market(m: dict, city: str) -> dict | None:
    """Parse a raw Gamma market dict into a normalised STORM-X market dict."""
    import json as _json
    question = m.get("question", "")
    if not question:
        return None

    raw_tokens = m.get("clobTokenIds", "[]")
    token_ids  = _json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
    raw_prices = m.get("outcomePrices", "[]")
    prices     = _json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices

    if len(token_ids) < 2 or len(prices) < 2:
        return None

    t_match = re.search(r"be\s+(-?\d+(?:\.\d+)?)°?C", question, re.IGNORECASE)
    if not t_match:
        return None

    temp_val       = float(t_match.group(1))
    is_lower_bound = bool(re.search(r"or\s+below|or\s+lower", question, re.IGNORECASE))
    is_upper_bound = bool(re.search(r"or\s+higher|or\s+above", question, re.IGNORECASE))

    return {
        "city":           city,
        "question":       question,
        "condition_id":   m.get("conditionId", ""),
        "token_yes":      token_ids[0],
        "token_no":       token_ids[1],
        "yes_price":      float(prices[0]),
        "no_price":       float(prices[1]),
        "volume_24h":     float(m.get("volumeNum", m.get("volume", 0)) or 0),
        "active":         bool(m.get("active", True)),
        "temp_value":     temp_val,
        "is_lower_bound": is_lower_bound,
        "is_upper_bound": is_upper_bound,
    }


async def fetch_markets(city: str, target_date: datetime | None = None) -> list[dict]:
    """Fetch all active temperature brackets for city on target_date.

    Args:
        city:        'london' or 'berlin'
        target_date: Defaults to tomorrow UTC.

    Returns:
        List of normalised market dicts, sorted by temp_value ascending.
        Empty list if the event is not live yet or the API is unreachable.
    """
    if target_date is None:
        target_date = datetime.now(timezone.utc) + timedelta(days=1)

    slug = _build_slug(city, target_date)
    logger.info("Fetching markets | city={} slug={}", city, slug)

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await client.get(f"{_GAMMA_BASE}/events", params={"slug": slug})
                resp.raise_for_status()
                events = resp.json()
                break
            except Exception as exc:
                if attempt == _MAX_RETRIES:
                    logger.error("Gamma API failed after {} attempts: {}", _MAX_RETRIES, exc)
                    return []
                import asyncio
                await asyncio.sleep(2 ** attempt)

    if not events:
        logger.info("No event found for slug: {}", slug)
        return []

    raw_markets = events[0].get("markets", [])
    markets = [
        parsed for m in raw_markets
        if m.get("active", True)
        and (parsed := _parse_market(m, city)) is not None
    ]
    markets.sort(key=lambda x: x["temp_value"])
    logger.info("Found {} active brackets for {} on {}",
                len(markets), city, target_date.strftime("%Y-%m-%d"))
    return markets
