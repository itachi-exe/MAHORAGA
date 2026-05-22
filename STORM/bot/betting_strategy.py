"""
Betting strategy — one bet per day, targeting the best available bracket.

Selection priority:
  1. Exact bracket  — non-boundary market whose temp_value is within ±1°C
                      of round(forecast).  e.g. forecast=19.7°C → 20°C.
  2. Closest boundary fallback — if no exact bracket exists on Polymarket:
       • "X°C or below"  where X is the lowest boundary ABOVE the forecast
         (model predicts temp will be ≤ X → YES bet)
       • "X°C or above"  where X is the highest boundary BELOW the forecast
         (model predicts temp will be ≥ X → YES bet)
     Both candidates are scored; the one with higher edge wins.

In all cases only one bet is placed — the full daily budget on the
single selected market.  If no market has edge > MIN_EDGE the day is skipped.
"""
import math
import logging
from config import MIN_EDGE, DAILY_BUDGET_USDC
from polymarket_client import place_order

MIN_ORDER_USDC = 1.0
MIN_SHARE_SIZE = 5.0

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
    if price <= 0 or price >= 1:
        return 0.0
    b = (1 - price) / price
    return max((prob * b - (1 - prob)) / b, 0.0)


def _score_market(market: dict, mu: float, sigma: float) -> tuple[str, float, float]:
    """Return (side, edge, bet_price) for the best side of a market."""
    model_prob   = _prob_for_market(market, mu, sigma)
    market_price = market["yes_price"]
    edge_yes     = model_prob - market_price
    edge_no      = -edge_yes  # edge on NO side

    if edge_yes >= edge_no:
        return "YES", edge_yes, market_price
    else:
        return "NO",  edge_no,  1 - market_price


def _select_target(markets: list[dict], mu: float) -> dict | None:
    """
    Priority 1: exact non-boundary bracket within ±1°C of round(forecast).
    Priority 2: closest boundary — "or below" just above forecast OR
                                   "or above" just below forecast.
    Returns the chosen market dict, or None if nothing suitable found.
    """
    target_temp = round(mu)

    # ── Priority 1: exact bracket ─────────────────────────────────────────
    exact = [m for m in markets if not m["is_lower_bound"] and not m["is_upper_bound"]]
    if exact:
        best_exact = min(exact, key=lambda m: abs(m["temp_value"] - target_temp))
        if abs(best_exact["temp_value"] - target_temp) <= 1:
            log.info("Exact bracket found: %.1f°C", best_exact["temp_value"])
            return best_exact

    log.info("No exact bracket within ±1°C of %d°C — trying boundary fallback", target_temp)

    # ── Priority 2a: "X or below" where X is the lowest value > forecast ─
    # Model says temp ≈ mu, so if X > mu the YES side is favoured
    below_candidates = [m for m in markets if m["is_lower_bound"] and m["temp_value"] >= mu]
    best_below = min(below_candidates, key=lambda m: m["temp_value"]) if below_candidates else None

    # ── Priority 2b: "X or above" where X is the highest value < forecast ─
    # Model says temp ≈ mu, so if X < mu the YES side is favoured
    above_candidates = [m for m in markets if m["is_upper_bound"] and m["temp_value"] <= mu]
    best_above = max(above_candidates, key=lambda m: m["temp_value"]) if above_candidates else None

    if best_below and best_above:
        # Pick whichever boundary is closer to the forecast (more uncertainty = more edge potential)
        dist_below = abs(best_below["temp_value"] - mu)
        dist_above = abs(best_above["temp_value"] - mu)
        chosen = best_below if dist_below <= dist_above else best_above
    elif best_below:
        chosen = best_below
    elif best_above:
        chosen = best_above
    else:
        log.warning("No suitable boundary market found for forecast %.2f°C", mu)
        return None

    log.info(
        "Boundary fallback: '%s' (%.1f°C, %s)",
        chosen["question"], chosen["temp_value"],
        "or below" if chosen["is_lower_bound"] else "or above",
    )
    return chosen


def evaluate_markets(markets: list[dict], forecast: dict, dry_run: bool = True) -> list[dict]:
    mu    = forecast["temperature_2m"]
    low   = forecast["temp_low"]
    high  = forecast["temp_high"]
    sigma = max((high - low) / (2 * 1.282), 0.5)

    log.info("Gaussian: μ=%.2f°C  σ=%.2f°C  range=[%.2f, %.2f]", mu, sigma, low, high)
    log.info("Forecast %.2f°C → target bracket %d°C", mu, round(mu))

    # ── Select the best available market ─────────────────────────────────
    target = _select_target(markets, mu)
    if target is None:
        return _all_skip(markets, mu, sigma)

    side, edge, bet_price = _score_market(target, mu, sigma)
    model_prob   = _prob_for_market(target, mu, sigma)
    market_price = target["yes_price"]

    if abs(edge) <= MIN_EDGE:
        log.info(
            "SKIP | %.1f°C | model=%.3f mkt=%.3f edge=%+.3f | below MIN_EDGE=%.2f",
            target["temp_value"], model_prob, market_price, edge, MIN_EDGE,
        )
        return _all_skip(markets, mu, sigma)

    # ── Validate Polymarket limits ────────────────────────────────────────
    usdc     = round(DAILY_BUDGET_USDC, 4)
    token_id = target["token_yes"] if side == "YES" else target["token_no"]

    if bet_price < 0.001 or bet_price > 0.999:
        log.info("SKIP %-3s | %.1f°C | price=%.4f outside [0.001, 0.999]",
                 side, target["temp_value"], bet_price)
        return _all_skip(markets, mu, sigma)

    shares = usdc / bet_price
    if shares < MIN_SHARE_SIZE:
        log.info("SKIP %-3s | %.1f°C | %.1f shares < min %g (need $%.2f)",
                 side, target["temp_value"], shares, MIN_SHARE_SIZE,
                 round(MIN_SHARE_SIZE * bet_price, 2))
        return _all_skip(markets, mu, sigma)

    # ── Place the bet ─────────────────────────────────────────────────────
    result = place_order(token_id, "BUY", bet_price, usdc, dry_run=dry_run)
    log.info(
        "BUY %-3s | %.1f°C | model=%.3f mkt=%.3f edge=%+.3f | $%.4f | %s",
        side, target["temp_value"], model_prob, market_price, edge, usdc,
        "DRY" if dry_run else "LIVE",
    )

    records = [{
        "condition_id": target["condition_id"],
        "question":     target["question"],
        "temp_value":   target["temp_value"],
        "action":       f"BUY {side}",
        "model_prob":   round(model_prob, 4),
        "market_price": round(market_price, 4),
        "edge":         round(edge, 4),
        "bet_usdc":     usdc,
        "order":        result,
    }]

    # Log all other markets as SKIP
    bet_temps = {r["temp_value"] for r in records}
    for m in markets:
        if m["temp_value"] not in bet_temps:
            mp = _prob_for_market(m, mu, sigma)
            records.append({
                "question":     m["question"],
                "temp_value":   m["temp_value"],
                "action":       "SKIP",
                "model_prob":   round(mp, 4),
                "market_price": round(m["yes_price"], 4),
                "edge":         round(mp - m["yes_price"], 4),
                "bet_usdc":     0.0,
            })

    records.sort(key=lambda r: r["temp_value"])
    log.info("Bets placed: 1 | Spent: $%.4f | Budget: $%.2f", usdc, DAILY_BUDGET_USDC)
    return records


def _all_skip(markets: list[dict], mu: float, sigma: float) -> list[dict]:
    records = [{
        "question":     m["question"],
        "temp_value":   m["temp_value"],
        "action":       "SKIP",
        "model_prob":   round(_prob_for_market(m, mu, sigma), 4),
        "market_price": round(m["yes_price"], 4),
        "edge":         round(_prob_for_market(m, mu, sigma) - m["yes_price"], 4),
        "bet_usdc":     0.0,
    } for m in markets]
    records.sort(key=lambda r: r["temp_value"])
    log.info("Bets placed: 0 | All markets skipped")
    return records
