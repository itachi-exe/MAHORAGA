"""
Polymarket client — London temperature markets via Gamma API event slug.

Slug pattern (predictable, changes daily):
  highest-temperature-in-london-on-may-16-2026
  highest-temperature-in-london-on-june-1-2026

One API call fetches the whole event with all temperature brackets + prices.
"""
import logging
import re
import requests
from datetime import datetime, timezone, timedelta

from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import ApiCreds, OrderArgs
from py_clob_client_v2.order_builder.constants import BUY, SELL

from config import (
    GAMMA_BASE, CLOB_BASE,
    POLY_API_KEY, POLY_API_SECRET, POLY_API_PASSPHRASE,
    POLY_PRIVATE_KEY, POLY_FUNDER, POLY_CHAIN_ID,
)

log = logging.getLogger(__name__)


def _build_slug(date: datetime) -> str:
    """highest-temperature-in-london-on-may-16-2026"""
    month = date.strftime("%B").lower()       # "may"
    day   = str(date.day)                     # "16"  (no leading zero)
    year  = str(date.year)                    # "2026"
    return f"highest-temperature-in-london-on-{month}-{day}-{year}"


def fetch_london_markets(target_date: datetime | None = None) -> list[dict]:
    """
    Fetch all active London highest-temperature markets for target_date
    (defaults to tomorrow UTC) via the Gamma API event slug.

    Returns list of dicts:
      question, condition_id, token_yes, token_no,
      yes_price, temp_value, is_lower_bound, is_upper_bound
    """
    if target_date is None:
        target_date = datetime.now(timezone.utc) + timedelta(days=1)

    slug = _build_slug(target_date)
    log.info("Fetching event slug: %s", slug)

    try:
        resp = requests.get(
            f"{GAMMA_BASE}/events",
            params={"slug": slug},
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as exc:
        log.error("Gamma API error: %s", exc)
        return []

    if not events:
        log.info("No event found for slug: %s", slug)
        return []

    markets_raw = events[0].get("markets", [])
    markets     = []

    for m in markets_raw:
        if not m.get("active", True):
            continue

        question    = m.get("question", "")
        condition_id = m.get("conditionId", "")
        import json as _json
        _raw_tokens = m.get("clobTokenIds", "[]")
        token_ids   = _json.loads(_raw_tokens) if isinstance(_raw_tokens, str) else _raw_tokens
        _raw_prices = m.get("outcomePrices", "[]")
        prices      = _json.loads(_raw_prices) if isinstance(_raw_prices, str) else _raw_prices

        if len(token_ids) < 2 or len(prices) < 2:
            continue

        token_yes  = token_ids[0]
        token_no   = token_ids[1]
        yes_price  = float(prices[0])

        # Parse temperature value from question
        t_match = re.search(r"be\s+(-?\d+(?:\.\d+)?)°?C", question, re.IGNORECASE)
        if not t_match:
            continue
        temp_val       = float(t_match.group(1))
        is_lower_bound = bool(re.search(r"or\s+below|or\s+lower", question, re.IGNORECASE))
        is_upper_bound = bool(re.search(r"or\s+higher|or\s+above", question, re.IGNORECASE))

        markets.append({
            "question":       question,
            "condition_id":   condition_id,
            "token_yes":      token_yes,
            "token_no":       token_no,
            "yes_price":      yes_price,
            "temp_value":     temp_val,
            "is_lower_bound": is_lower_bound,
            "is_upper_bound": is_upper_bound,
        })

    markets.sort(key=lambda x: x["temp_value"])
    log.info("Found %d markets for %s", len(markets), slug)
    return markets


def _make_client() -> ClobClient:
    return ClobClient(
        host=CLOB_BASE,
        chain_id=POLY_CHAIN_ID,
        key=POLY_PRIVATE_KEY,
        creds=ApiCreds(POLY_API_KEY, POLY_API_SECRET, POLY_API_PASSPHRASE),
        signature_type=1,     # POLY_PROXY — EOA signs on behalf of proxy/funder wallet
        funder=POLY_FUNDER,
    )


def place_order(token_id: str, side: str, price: float, usdc_amount: float, dry_run: bool = True) -> dict:
    info = {"token_id": token_id[:20] + "...", "side": side, "usdc": round(usdc_amount, 4)}
    if dry_run:
        log.info("[DRY-RUN] %s", info)
        return {"dry_run": True, **info}
    try:
        client = _make_client()

        # Hard Polymarket limits — must be enforced before tick-size clamp
        price = max(min(price, 0.999), 0.001)

        # Clamp price to valid tick-size range [tick, 1-tick]
        tick = float(client.get_tick_size(token_id))
        price = max(price, tick)
        price = min(price, 1.0 - tick)

        # size = shares to buy; for BUY: cost = size * price in USDC
        size = round(usdc_amount / price, 2)

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
        )
        result = client.create_and_post_order(order_args)
        log.info("Order placed: %s", result)
        return result
    except Exception as exc:
        log.error("Order failed: %s", exc)
        return {"error": str(exc), **info}
