"""
igris_backbone.py
─────────────────
Pure-Python port of the IGRIS 4-signal consensus gate.
Drop-in replacement for the former PyO3 Rust extension — identical function
signature, identical thresholds, identical return structure.

Usage:
    from igris_backbone import backbone_signal
"""

# ── Signal thresholds ────────────────────────────────────────────────────────
_MOMENTUM_UP    =  0.05   # % price change over 3 closes → UP
_MOMENTUM_DOWN  = -0.05
_OB_IMBAL_UP    =  0.15   # (bids-asks)/(bids+asks) → UP
_OB_IMBAL_DOWN  = -0.15
_FUND_BEARISH   =  0.01   # funding rate (decimal) triggers bearish divergence
_FUND_BULLISH   = -0.01
_OI_BEARISH     = -0.5    # % OI change triggers bearish divergence
_OI_BULLISH     =  0.5
_VEL_UP         =  0.03   # odds shift (odds[-1] - odds[-3]) → UP
_VEL_DOWN       = -0.03


def _signal_momentum(closes: list) -> tuple:
    """SIGNAL 1 — (last_close - close_3_bars_ago) / close_3_bars_ago * 100"""
    if len(closes) < 4:
        return "NONE", 0.0
    anchor = float(closes[-4])
    if anchor == 0.0:
        return "NONE", 0.0
    momentum = (float(closes[-1]) - anchor) / anchor * 100.0
    if momentum > _MOMENTUM_UP:
        return "UP", momentum
    if momentum < _MOMENTUM_DOWN:
        return "DOWN", momentum
    return "NONE", momentum


def _signal_ob_imbalance(bids: float, asks: float) -> tuple:
    """SIGNAL 2 — (bids - asks) / (bids + asks)"""
    total = bids + asks
    if total == 0.0:
        return "NONE", 0.0
    imbalance = (bids - asks) / total
    if imbalance > _OB_IMBAL_UP:
        return "UP", imbalance
    if imbalance < _OB_IMBAL_DOWN:
        return "DOWN", imbalance
    return "NONE", imbalance


def _signal_funding_oi(funding: float, oi_change: float) -> tuple:
    """SIGNAL 3 — Funding rate + OI divergence"""
    abs_oi = abs(oi_change)
    score  = (funding / abs_oi * (1.0 if oi_change >= 0 else -1.0)) if abs_oi > 0.0 else 0.0
    if funding > _FUND_BEARISH and oi_change < _OI_BEARISH:
        return "DOWN", score
    if funding < _FUND_BULLISH and oi_change > _OI_BULLISH:
        return "UP", score
    return "NONE", score


def _signal_odds_velocity(odds: list) -> tuple:
    """SIGNAL 4 — odds[-1] - odds[-3]  (sharp money detection)"""
    if len(odds) < 3:
        return "NONE", 0.0
    velocity = float(odds[-1]) - float(odds[-3])
    if velocity > _VEL_UP:
        return "UP", velocity
    if velocity < _VEL_DOWN:
        return "DOWN", velocity
    return "NONE", velocity


def backbone_signal(data: dict) -> dict:
    """
    Run 4 independent market signals and return a consensus gate result.
    All 4 must agree on the same direction or approved=False.

    Input dict keys:
        closes        list[float]   last 15 one-minute BTC closes
        bids_volume   float         total bid depth (Bybit L2 top-50)
        asks_volume   float         total ask depth (Bybit L2 top-50)
        funding_rate  float         current Bybit perpetual funding rate
        oi_change_pct float         % change in open interest over last 5 min
        odds_history  list[float]   last ≤10 YES-token mid-price snapshots

    Returns:
        {
            "approved":  bool,
            "direction": "UP" | "DOWN" | "NONE",
            "scores": {
                "momentum":           float,
                "ob_imbalance":       float,
                "funding_divergence": float,
                "odds_velocity":      float,
            },
            "reason": str,   # empty string when approved=True
        }
    """
    closes      = data.get("closes", [])
    bids_vol    = float(data.get("bids_volume", 0.0))
    asks_vol    = float(data.get("asks_volume", 0.0))
    funding     = float(data.get("funding_rate", 0.0))
    oi_change   = float(data.get("oi_change_pct", 0.0))
    odds_hist   = data.get("odds_history", [])

    s1_dir, s1_score = _signal_momentum(closes)
    s2_dir, s2_score = _signal_ob_imbalance(bids_vol, asks_vol)
    s3_dir, s3_score = _signal_funding_oi(funding, oi_change)
    s4_dir, s4_score = _signal_odds_velocity(odds_hist)

    scores = {
        "momentum":           s1_score,
        "ob_imbalance":       s2_score,
        "funding_divergence": s3_score,
        "odds_velocity":      s4_score,
    }

    signals = [(1, s1_dir), (2, s2_dir), (3, s3_dir), (4, s4_dir)]

    # Hard gate: any NONE → reject immediately
    for n, direction in signals:
        if direction == "NONE":
            return {
                "approved":  False,
                "direction": "NONE",
                "scores":    scores,
                "reason":    f"signal {n} inconclusive",
            }

    # All signals active — check consensus
    dirs = {d for _, d in signals}
    if len(dirs) == 1:
        return {
            "approved":  True,
            "direction": dirs.pop(),
            "scores":    scores,
            "reason":    "",
        }

    # Conflict
    detail = ", ".join(f"s{n}={d}" for n, d in signals)
    return {
        "approved":  False,
        "direction": "NONE",
        "scores":    scores,
        "reason":    f"signal conflict: {detail}",
    }
