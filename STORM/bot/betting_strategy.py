"""
Betting strategy — places the daily budget on the best edge market.

Polymarket CLOB requires a minimum of $1 per order, so with a $1 daily
budget we bet it all on the single highest-Kelly candidate. If budget
allows multiple $1+ bets, it distributes proportionally by Kelly fraction
(each bet rounded up to the $1 minimum, capped at total budget).
"""
import math
import logging
from config import MIN_EDGE, DAILY_BUDGET_USDC
from polymarket_client import place_order

MIN_ORDER_USDC  = 1.0   # Polymarket CLOB minimum order in USDC
MIN_SHARE_SIZE  = 5.0   # Polymarket CLOB minimum shares per order (hard API limit)

log = logging.getLogger(__name__)


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def _prob_for_market(market: dict, mu: float, sigma: float) -> float:
    t  = market["temp_value"]
    lo = market["is_lower_bound"]
    hi = market["is_upper_bound"]
    if lo:
        return _normal_cdf(t + 0.5, mu, sigma)
    elif hi:
        return 1 - _normal_cdf(t - 0.5, mu, sigma)
    else:
        return _normal_cdf(t + 0.5, mu, sigma) - _normal_cdf(t - 0.5, mu, sigma)


def _kelly_fraction(prob: float, price: float) -> float:
    """Raw Kelly fraction (not yet in USDC)."""
    if price <= 0 or price >= 1:
        return 0.0
    b = (1 - price) / price
    f = (prob * b - (1 - prob)) / b
    return max(f, 0.0)


def evaluate_markets(markets: list[dict], forecast: dict, dry_run: bool = True) -> list[dict]:
    mu    = forecast["temperature_2m"]
    low   = forecast["temp_low"]
    high  = forecast["temp_high"]
    sigma = max((high - low) / (2 * 1.282), 0.5)

    log.info("Gaussian: μ=%.2f°C  σ=%.2f°C  range=[%.2f, %.2f]", mu, sigma, low, high)

    # ── Pass 1: score every market ────────────────────────────────────────
    candidates = []
    for market in markets:
        model_prob   = _prob_for_market(market, mu, sigma)
        market_price = market["yes_price"]
        edge         = model_prob - market_price

        if edge > MIN_EDGE:
            # YES is underpriced
            kelly = _kelly_fraction(model_prob, market_price)
            candidates.append((market, "YES", model_prob, market_price, edge, kelly))
        elif -edge > MIN_EDGE:
            # NO is underpriced — store positive NO-side edge
            no_prob  = 1 - model_prob
            no_price = 1 - market_price
            no_edge  = -edge          # flip sign: actual advantage on NO side
            kelly    = _kelly_fraction(no_prob, no_price)
            candidates.append((market, "NO", model_prob, market_price, no_edge, kelly))

    # ── Pass 2: pick ONLY the single best candidate ───────────────────────
    # Multiple correlated temperature brackets can all show edge; betting
    # on more than one would over-expose the model to the same outcome.
    candidates.sort(key=lambda c: c[5], reverse=True)
    candidates = candidates[:1]   # one bet per day, highest Kelly only
    records = []
    budget_remaining = DAILY_BUDGET_USDC

    for market, side, model_prob, market_price, edge, kelly in candidates:
        if budget_remaining < MIN_ORDER_USDC:
            break

        # Bet the full remaining daily budget on the best market
        usdc = round(budget_remaining, 4)

        token_id  = market["token_yes"] if side == "YES" else market["token_no"]
        bet_price = market_price if side == "YES" else (1 - market_price)

        # Polymarket hard limits: price must be in [0.001, 0.999] and shares >= 5
        if bet_price < 0.001 or bet_price > 0.999:
            log.info(
                "SKIP %-3s | %5.1f°C | price=%.4f out of Polymarket range [0.001, 0.999]",
                side, market["temp_value"], bet_price,
            )
            continue

        shares = usdc / bet_price
        if shares < MIN_SHARE_SIZE:
            min_cost = round(MIN_SHARE_SIZE * bet_price, 2)
            log.info(
                "SKIP %-3s | %5.1f°C | price=%.4f → %.1f shares < min %g (need $%.2f budget)",
                side, market["temp_value"], bet_price, shares, MIN_SHARE_SIZE, min_cost,
            )
            continue

        budget_remaining -= usdc
        result = place_order(token_id, "BUY", bet_price, usdc, dry_run=dry_run)

        log.info(
            "BUY %-3s | %5.1f°C | model=%.3f mkt=%.3f edge=%+.3f | $%.4f | %s",
            side, market["temp_value"], model_prob, market_price, edge, usdc,
            "DRY" if dry_run else "LIVE",
        )

        records.append({
            "condition_id": market["condition_id"],
            "question":     market["question"],
            "temp_value":   market["temp_value"],
            "action":       f"BUY {side}",
            "model_prob":   round(model_prob, 4),
            "market_price": round(market_price, 4),
            "edge":         round(edge, 4),
            "bet_usdc":     usdc,
            "order":        result,
        })

    # Skipped markets (no edge)
    bet_temps = {r["temp_value"] for r in records}
    for market in markets:
        if market["temp_value"] not in bet_temps:
            records.append({
                "question":     market["question"],
                "temp_value":   market["temp_value"],
                "action":       "SKIP",
                "model_prob":   round(_prob_for_market(market, mu, sigma), 4),
                "market_price": round(market["yes_price"], 4),
                "edge":         round(_prob_for_market(market, mu, sigma) - market["yes_price"], 4),
                "bet_usdc":     0.0,
            })

    records.sort(key=lambda r: r["temp_value"])
    placed = sum(1 for r in records if r["action"] != "SKIP")
    total  = sum(r["bet_usdc"] for r in records)
    log.info("Bets placed: %d | Total spent: $%.4f | Budget: $%.2f",
             placed, total, DAILY_BUDGET_USDC)
    return records
