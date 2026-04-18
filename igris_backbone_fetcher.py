"""
igris_backbone_fetcher.py
─────────────────────────
Fetches all market data needed to build the igris_backbone input dict.
Runs entirely via the `requests` library — no pybit, no websockets.

Background thread polls Polymarket YES-token odds every 30 seconds and
maintains a rolling deque(maxlen=10) so backbone_signal always has fresh
odds velocity data.

Usage:
    from igris_backbone_fetcher import build_backbone_input, start_odds_polling
    start_odds_polling()  # call once at startup
    data = build_backbone_input("UP")
    if data:
        from igris_backbone import backbone_signal
        result = backbone_signal(data)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import requests

# ── Module logger — uses same name hierarchy as PolymarketTrader ─────────────
log = logging.getLogger("PolymarketTrader.fetcher")

# ── Constants ────────────────────────────────────────────────────────────────
BYBIT_BASE          = "https://api.bybit.com/v5"
BYBIT_TIMEOUT       = 5         # seconds per request
BYBIT_RETRIES       = 1         # retry once on failure
GAMMA_API           = "https://gamma-api.polymarket.com"
CLOB_API            = "https://clob.polymarket.com"
ODDS_BUFFER_SIZE    = 10        # rolling window for odds snapshots
ODDS_POLL_INTERVAL  = 30        # seconds between Polymarket odds polls

# ── Shared odds rolling buffer ───────────────────────────────────────────────
_odds_deque: deque[float] = deque(maxlen=ODDS_BUFFER_SIZE)
_odds_lock   = threading.Lock()
_odds_thread: Optional[threading.Thread] = None
_odds_stop   = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None) -> Optional[dict]:
    """GET with timeout + one retry. Returns parsed JSON or None on failure."""
    for attempt in range(BYBIT_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=BYBIT_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt < BYBIT_RETRIES:
                log.debug(f"[Fetcher] {url} attempt {attempt + 1} failed ({exc}), retrying…")
            else:
                log.warning(
                    f"[Fetcher] {url} failed after {BYBIT_RETRIES + 1} attempts: {exc}"
                )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public fetchers
# ─────────────────────────────────────────────────────────────────────────────

def get_btc_closes(n: int = 15) -> list[float]:
    """
    Fetch last n one-minute BTC/USDT close prices from Bybit.
    Returns oldest→newest. Returns [] on failure.
    """
    data = _get(
        f"{BYBIT_BASE}/market/kline",
        params={"category": "linear", "symbol": "BTCUSDT", "interval": "1", "limit": n},
    )
    if not data or data.get("retCode") != 0:
        log.warning("[Fetcher] get_btc_closes: bad response")
        return []
    try:
        # Bybit returns newest-first; each row is [openTime, open, high, low, close, volume, …]
        rows = data["result"]["list"]
        closes = [float(row[4]) for row in reversed(rows)]
        return closes
    except Exception as exc:
        log.warning(f"[Fetcher] get_btc_closes parse error: {exc}")
        return []


def get_orderbook_volumes() -> tuple[float, float]:
    """
    Fetch Bybit L2 top-50 order book for BTCUSDT.
    Returns (bids_volume, asks_volume). Returns (0.0, 0.0) on failure.
    """
    data = _get(
        f"{BYBIT_BASE}/market/orderbook",
        params={"category": "linear", "symbol": "BTCUSDT", "limit": 50},
    )
    if not data or data.get("retCode") != 0:
        log.warning("[Fetcher] get_orderbook_volumes: bad response")
        return 0.0, 0.0
    try:
        result = data["result"]
        bids_vol = sum(float(entry[1]) for entry in result.get("b", []))
        asks_vol = sum(float(entry[1]) for entry in result.get("a", []))
        return bids_vol, asks_vol
    except Exception as exc:
        log.warning(f"[Fetcher] get_orderbook_volumes parse error: {exc}")
        return 0.0, 0.0


def get_funding_and_oi() -> tuple[float, float]:
    """
    Fetch Bybit funding rate + 5-min OI % change for BTCUSDT.
    Returns (funding_rate, oi_change_pct). Returns (0.0, 0.0) on failure.
    """
    # Funding rate from tickers
    funding_rate = 0.0
    ticker_data = _get(
        f"{BYBIT_BASE}/market/tickers",
        params={"category": "linear", "symbol": "BTCUSDT"},
    )
    if ticker_data and ticker_data.get("retCode") == 0:
        try:
            items = ticker_data["result"].get("list", [])
            if items:
                funding_rate = float(items[0].get("fundingRate", 0))
        except Exception as exc:
            log.warning(f"[Fetcher] funding rate parse error: {exc}")

    # OI change from open-interest endpoint (2 snapshots)
    oi_change_pct = 0.0
    oi_data = _get(
        f"{BYBIT_BASE}/market/open-interest",
        params={"category": "linear", "symbol": "BTCUSDT", "intervalTime": "5min", "limit": 2},
    )
    if oi_data and oi_data.get("retCode") == 0:
        try:
            rows = oi_data["result"].get("list", [])
            if len(rows) >= 2:
                # Bybit returns newest-first
                oi_latest = float(rows[0]["openInterest"])
                oi_prev   = float(rows[1]["openInterest"])
                if oi_prev != 0.0:
                    oi_change_pct = (oi_latest - oi_prev) / oi_prev * 100.0
        except Exception as exc:
            log.warning(f"[Fetcher] OI change parse error: {exc}")

    return funding_rate, oi_change_pct


def get_active_btc_market_id() -> Optional[str]:
    """
    Find the condition_id (used by CLOB API) for the active BTC Up/Down
    15-minute Polymarket market whose end_date is the nearest future 15-min
    boundary.  Returns None if not found.
    """
    data = _get(
        f"{GAMMA_API}/markets",
        params={"tag": "bitcoin", "active": "true", "limit": 10},
    )
    if not data:
        return None

    # Gamma returns a list directly or wrapped in a "data" key depending on version
    markets = data if isinstance(data, list) else (data.get("data") or data.get("markets") or [])

    now_ts = datetime.now(timezone.utc).timestamp()
    best_market = None
    best_delta  = float("inf")

    for m in markets:
        question = m.get("question", "")
        if "up or down" not in question.lower():
            continue
        end_str = m.get("endDateIso") or m.get("end_date_iso") or m.get("endDate", "")
        if not end_str:
            continue
        try:
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            delta  = end_dt.timestamp() - now_ts
            if 0 < delta < best_delta:
                best_delta  = delta
                best_market = m
        except Exception:
            continue

    if best_market:
        return (
            best_market.get("conditionId")
            or best_market.get("condition_id")
            or best_market.get("clob_token_ids", [None])[0]
        )
    return None


def _fetch_yes_odds(condition_id: str) -> Optional[float]:
    """
    Fetch the current YES-token mid price from Polymarket CLOB.
    Returns a float in [0, 1] or None on failure.
    """
    data = _get(f"{CLOB_API}/markets/{condition_id}")
    if not data:
        return None
    try:
        tokens = data.get("tokens", [])
        for tok in tokens:
            if str(tok.get("outcome", "")).upper() == "YES":
                price = float(tok.get("price", 0))
                if 0.0 < price < 1.0:
                    return price
        # Fallback: use first token price
        if tokens:
            price = float(tokens[0].get("price", 0))
            if 0.0 < price < 1.0:
                return price
    except Exception as exc:
        log.debug(f"[Fetcher] _fetch_yes_odds parse error: {exc}")
    return None


def get_odds_history() -> list[float]:
    """
    Return current rolling odds buffer (oldest→newest).
    May have fewer than 10 entries at startup — backbone will return
    signal 4 = NONE in that case (< 3 entries minimum required).
    """
    with _odds_lock:
        return list(_odds_deque)


# ─────────────────────────────────────────────────────────────────────────────
# Background odds polling thread
# ─────────────────────────────────────────────────────────────────────────────

def _odds_poll_worker() -> None:
    log.info("[Fetcher] Odds polling thread started")
    while not _odds_stop.is_set():
        try:
            condition_id = get_active_btc_market_id()
            if condition_id:
                price = _fetch_yes_odds(condition_id)
                if price is not None:
                    with _odds_lock:
                        _odds_deque.append(price)
                    log.debug(f"[Fetcher] Odds snapshot: {price:.4f} (buffer={len(_odds_deque)})")
            else:
                log.debug("[Fetcher] No active BTC market found this poll")
        except Exception as exc:
            log.warning(f"[Fetcher] Odds poll error: {exc}")
        _odds_stop.wait(ODDS_POLL_INTERVAL)
    log.info("[Fetcher] Odds polling thread stopped")


def start_odds_polling() -> None:
    """
    Start the background odds polling thread. Safe to call multiple times
    — only starts a new thread if one is not already running.
    """
    global _odds_thread
    if _odds_thread is not None and _odds_thread.is_alive():
        return
    _odds_stop.clear()
    _odds_thread = threading.Thread(
        target=_odds_poll_worker,
        name="igris-odds-poll",
        daemon=True,
    )
    _odds_thread.start()


def stop_odds_polling() -> None:
    """Signal the background polling thread to stop cleanly."""
    _odds_stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# Main assembler
# ─────────────────────────────────────────────────────────────────────────────

def build_backbone_input(direction: str) -> Optional[dict]:
    """
    Fetch all four data sources and assemble the backbone_signal input dict.
    Returns None if any critical data is missing (closes or order book)
    so that the calling code can skip the bet cleanly.

    Args:
        direction: "UP" or "DOWN" (from upstream MLP)

    Returns:
        dict with keys: closes, bids_volume, asks_volume, funding_rate,
                        oi_change_pct, odds_history
        or None on critical failure.
    """
    ts = datetime.now(timezone.utc).isoformat()

    closes = get_btc_closes()
    if len(closes) < 4:
        log.warning(f"[Fetcher] {ts} | build_backbone_input: not enough closes ({len(closes)}) — skip")
        return None

    bids_vol, asks_vol = get_orderbook_volumes()
    if bids_vol == 0.0 and asks_vol == 0.0:
        log.warning(f"[Fetcher] {ts} | build_backbone_input: order book empty — skip")
        return None

    funding_rate, oi_change_pct = get_funding_and_oi()

    odds_hist = get_odds_history()
    if not odds_hist:
        log.debug(f"[Fetcher] {ts} | odds buffer empty — signal 4 will be NONE")

    log.debug(
        f"[Fetcher] {ts} | backbone input assembled "
        f"| dir={direction} closes={len(closes)} "
        f"bids={bids_vol:.1f} asks={asks_vol:.1f} "
        f"funding={funding_rate:.6f} oi={oi_change_pct:.3f}% "
        f"odds_hist={len(odds_hist)}"
    )

    return {
        "closes":        closes,
        "bids_volume":   bids_vol,
        "asks_volume":   asks_vol,
        "funding_rate":  funding_rate,
        "oi_change_pct": oi_change_pct,
        "odds_history":  odds_hist,
    }
