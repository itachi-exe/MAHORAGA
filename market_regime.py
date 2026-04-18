"""
market_regime.py
────────────────
Shared multi-timeframe regime detection gate for MAHORAGA and IGRIS.

Usage:
    from market_regime import get_regime_verdict

    verdict = get_regime_verdict("UP")   # or "DOWN"
    if not verdict["approved"]:
        skip_trade(verdict["reason"])
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
import requests

# ── Logger ────────────────────────────────────────────────────────────────────
log = logging.getLogger("MAHORAGA.regime")

# ── Constants ─────────────────────────────────────────────────────────────────
CANDLES_15M             = 100
CANDLES_4H              = 10

ADX_PERIOD              = 14
BB_PERIOD               = 20
BB_STD                  = 2
ATR_PERIOD              = 14
ATR_LOOKBACK            = 20
BBW_LOOKBACK            = 20

ADX_CHOPPY_THRESHOLD    = 15
ADX_TRENDING_THRESHOLD  = 20
ADX_MODERATE_THRESHOLD  = 25
ADX_STRONG_THRESHOLD    = 35

ATR_RATIO_HIGH          = 1.2
ATR_RATIO_LOW           = 0.8
BBW_RATIO_HIGH          = 1.2
BBW_RATIO_LOW           = 0.8

CACHE_DURATION_SECONDS  = 300

_BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"

# ── Thread-safe cache ─────────────────────────────────────────────────────────
_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"ts": None, "data": None}


# ── Custom exception ──────────────────────────────────────────────────────────
class FetchError(RuntimeError):
    pass


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_candles(interval: str, limit: int) -> list[dict]:
    """
    Fetch BTCUSDT klines from Bybit REST API.
    Returns list of OHLCV dicts ordered oldest → newest.
    Retries once on failure; raises FetchError on second failure.
    """
    params = {
        "category": "linear",
        "symbol":   "BTCUSDT",
        "interval": interval,
        "limit":    limit,
    }
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            resp = requests.get(_BYBIT_KLINE_URL, params=params, timeout=5)
            resp.raise_for_status()
            payload = resp.json()
            raw = payload.get("result", {}).get("list", [])
            if not raw:
                raise FetchError(f"Empty kline list from Bybit (interval={interval})")
            # Bybit returns newest-first: [ts, open, high, low, close, volume, turnover]
            candles = []
            for row in reversed(raw):
                candles.append({
                    "open":   float(row[1]),
                    "high":   float(row[2]),
                    "low":    float(row[3]),
                    "close":  float(row[4]),
                    "volume": float(row[5]),
                })
            return candles
        except FetchError:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                log.debug(f"[Regime] fetch attempt 1 failed ({exc}), retrying…")
    raise FetchError(f"Bybit fetch failed after 2 attempts: {last_exc}")


# ── Wilder smoothing ──────────────────────────────────────────────────────────

def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's running smoothing: out[i] = out[i-1] - out[i-1]/period + values[i]"""
    out = np.zeros(len(values))
    if len(values) < period:
        return out
    out[period - 1] = np.sum(values[:period])
    for i in range(period, len(values)):
        out[i] = out[i - 1] - out[i - 1] / period + values[i]
    return out


# ── Indicators ────────────────────────────────────────────────────────────────

def calc_adx(candles: list[dict]) -> tuple[float, float, float]:
    """
    ADX (Wilder, period=ADX_PERIOD).
    Returns (adx, plus_di, minus_di) as the most-recent values.
    """
    n = len(candles)
    if n < ADX_PERIOD + 1:
        return 0.0, 0.0, 0.0

    highs  = np.array([c["high"]  for c in candles], dtype=np.float64)
    lows   = np.array([c["low"]   for c in candles], dtype=np.float64)
    closes = np.array([c["close"] for c in candles], dtype=np.float64)

    tr   = np.zeros(n)
    pdm  = np.zeros(n)
    ndm  = np.zeros(n)

    for i in range(1, n):
        hl   = highs[i]  - lows[i]
        hpc  = abs(highs[i]  - closes[i - 1])
        lpc  = abs(lows[i]   - closes[i - 1])
        tr[i] = max(hl, hpc, lpc)

        up   = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        pdm[i] = up   if (up > down and up > 0)   else 0.0
        ndm[i] = down if (down > up and down > 0) else 0.0

    s_tr  = _wilder_smooth(tr,  ADX_PERIOD)
    s_pdm = _wilder_smooth(pdm, ADX_PERIOD)
    s_ndm = _wilder_smooth(ndm, ADX_PERIOD)

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = np.where(s_tr > 0, 100.0 * s_pdm / s_tr, 0.0)
        ndi = np.where(s_tr > 0, 100.0 * s_ndm / s_tr, 0.0)
        dx_num = np.abs(pdi - ndi)
        dx_den = pdi + ndi
        dx = np.where(dx_den > 0, 100.0 * dx_num / dx_den, 0.0)

    adx_arr = _wilder_smooth(dx, ADX_PERIOD)

    return float(adx_arr[-1]), float(pdi[-1]), float(ndi[-1])


def calc_atr_ratio(candles: list[dict]) -> float:
    """
    ATR ratio = current_ATR / mean(last ATR_LOOKBACK ATR values).
    Returns 1.0 if not enough data.
    """
    n = len(candles)
    if n < ATR_PERIOD + 1:
        return 1.0

    highs  = np.array([c["high"]  for c in candles], dtype=np.float64)
    lows   = np.array([c["low"]   for c in candles], dtype=np.float64)
    closes = np.array([c["close"] for c in candles], dtype=np.float64)

    tr = np.zeros(n)
    for i in range(1, n):
        hl  = highs[i] - lows[i]
        hpc = abs(highs[i]  - closes[i - 1])
        lpc = abs(lows[i]   - closes[i - 1])
        tr[i] = max(hl, hpc, lpc)

    atr = _wilder_smooth(tr, ATR_PERIOD)

    # Take only values where Wilder smoothing has warmed up
    valid = atr[ATR_PERIOD:]
    if len(valid) < 2:
        return 1.0

    lookback = valid[-ATR_LOOKBACK:] if len(valid) >= ATR_LOOKBACK else valid
    mean_atr = np.mean(lookback[:-1]) if len(lookback) > 1 else lookback[-1]
    if mean_atr == 0.0:
        return 1.0
    return float(valid[-1] / mean_atr)


def calc_bbw_ratio(candles: list[dict]) -> float:
    """
    Bollinger Band Width ratio = current_BBW / mean(last BBW_LOOKBACK BBW values).
    Returns 1.0 if not enough data.
    """
    if len(candles) < BB_PERIOD:
        return 1.0

    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    bbw = np.zeros(len(closes))

    for i in range(BB_PERIOD - 1, len(closes)):
        window = closes[i - BB_PERIOD + 1 : i + 1]
        mid    = np.mean(window)
        std    = np.std(window, ddof=0)
        upper  = mid + BB_STD * std
        lower  = mid - BB_STD * std
        bbw[i] = (upper - lower) / mid if mid != 0.0 else 0.0

    valid = bbw[BB_PERIOD - 1:]
    if len(valid) < 2:
        return 1.0

    lookback = valid[-BBW_LOOKBACK:] if len(valid) >= BBW_LOOKBACK else valid
    mean_bbw = np.mean(lookback[:-1]) if len(lookback) > 1 else lookback[-1]
    if mean_bbw == 0.0:
        return 1.0
    return float(valid[-1] / mean_bbw)


# ── Regime classification ─────────────────────────────────────────────────────

def classify_regime(
    adx: float,
    plus_di: float,
    minus_di: float,
    atr_ratio: float,
    bbw_ratio: float,
    closes: list[float],
) -> str:
    """
    Classify market regime. Rules applied in strict priority order.
    """
    # 1. CHOPPY — no directional edge, low volatility
    if adx < ADX_CHOPPY_THRESHOLD and atr_ratio < ATR_RATIO_LOW:
        return "CHOPPY"

    # 2. RANGING — no trend, volatility may exist
    if adx < ADX_TRENDING_THRESHOLD:
        return "RANGING"

    # 3. TRENDING_UP
    if adx >= ADX_TRENDING_THRESHOLD and plus_di > minus_di:
        if len(closes) >= 4 and closes[-1] > closes[-4]:
            return "TRENDING_UP"

    # 4. TRENDING_DOWN
    if adx >= ADX_TRENDING_THRESHOLD and minus_di > plus_di:
        if len(closes) >= 4 and closes[-1] < closes[-4]:
            return "TRENDING_DOWN"

    # Trending ADX but price/DI inconclusive — treat as RANGING
    return "RANGING"


def _trend_strength(adx_4h: float) -> str:
    if adx_4h >= ADX_STRONG_THRESHOLD:
        return "STRONG"
    if adx_4h >= ADX_MODERATE_THRESHOLD:
        return "MODERATE"
    if adx_4h >= ADX_TRENDING_THRESHOLD:
        return "WEAK"
    return "NONE"


def _bias_from_regime(regime_4h: str) -> str:
    if regime_4h == "TRENDING_UP":
        return "UP"
    if regime_4h == "TRENDING_DOWN":
        return "DOWN"
    return "NEUTRAL"


# ── Approval logic ────────────────────────────────────────────────────────────

def _evaluate(
    proposed: str,
    regime_15m: str,
    regime_4h: str,
    adx_15m: float,
    adx_4h: float,
    plus_di_15m: float,
    minus_di_15m: float,
    plus_di_4h: float,
    minus_di_4h: float,
    atr_ratio_15m: float,
    atr_ratio_4h: float,
    bbw_ratio_15m: float,
    bbw_ratio_4h: float,
) -> dict:
    strength = _trend_strength(adx_4h)
    bias     = _bias_from_regime(regime_4h)

    base = {
        "regime_15m":    regime_15m,
        "regime_4h":     regime_4h,
        "adx_15m":       round(adx_15m,   2),
        "adx_4h":        round(adx_4h,    2),
        "plus_di_15m":   round(plus_di_15m,  2),
        "minus_di_15m":  round(minus_di_15m, 2),
        "plus_di_4h":    round(plus_di_4h,   2),
        "minus_di_4h":   round(minus_di_4h,  2),
        "atr_ratio_15m": round(atr_ratio_15m, 4),
        "atr_ratio_4h":  round(atr_ratio_4h,  4),
        "bbw_ratio_15m": round(bbw_ratio_15m, 4),
        "bbw_ratio_4h":  round(bbw_ratio_4h,  4),
        "trend_strength": strength,
        "bias":          bias,
        "alignment":     False,
        "approved":      False,
        "reason":        "",
        "cached":        False,
    }

    # ── CHOPPY check (immediate reject) ──────────────────────────────────────
    if regime_15m == "CHOPPY":
        base["reason"] = "choppy market on 15m — no edge"
        return base
    if regime_4h == "CHOPPY":
        base["reason"] = "choppy market on 4h — no edge"
        return base

    # ── Direction vs bias check ───────────────────────────────────────────────
    if bias != "NEUTRAL" and proposed != bias:
        base["reason"] = f"proposed {proposed} against 4h bias {bias}"
        return base

    # ── Alignment matrix ──────────────────────────────────────────────────────
    # Both ranging — allow either direction
    if regime_4h == "RANGING" and regime_15m == "RANGING":
        if adx_15m < ADX_TRENDING_THRESHOLD and adx_4h < ADX_TRENDING_THRESHOLD:
            base["alignment"] = True
            base["approved"]  = True
            base["reason"]    = "both timeframes ranging — fade extremes"
            return base

    # 4h ranging, 15m choppy
    if regime_4h == "RANGING" and regime_15m == "CHOPPY":
        base["reason"] = "15m choppy inside ranging 4h — skip"
        return base

    # 4h trending up
    if regime_4h == "TRENDING_UP":
        if regime_15m in ("TRENDING_UP", "RANGING"):
            base["alignment"] = True
        else:
            base["reason"] = "15m fighting 4h trend — conflict"
            return base

    # 4h trending down
    elif regime_4h == "TRENDING_DOWN":
        if regime_15m in ("TRENDING_DOWN", "RANGING"):
            base["alignment"] = True
        else:
            base["reason"] = "15m fighting 4h trend — conflict"
            return base

    # 4h ranging + 15m trending — ok but note it
    elif regime_4h == "RANGING":
        base["alignment"] = True

    # ── ADX confirmation — at least one TF must be trending ──────────────────
    if adx_15m < ADX_TRENDING_THRESHOLD and adx_4h < ADX_TRENDING_THRESHOLD:
        base["reason"] = "neither timeframe trending (ADX too low)"
        return base

    base["approved"] = True
    return base


# ── Core public function ──────────────────────────────────────────────────────

def get_regime_verdict(proposed_direction: str) -> dict:
    """
    Fetch BTC market regime on 15m and 4h and gate the proposed trade direction.

    Args:
        proposed_direction: "UP" or "DOWN"

    Returns:
        Full verdict dict (see module docstring for schema).
        On fetch failure returns approved=False, reason="regime data unavailable".
    """
    now = time.monotonic()

    # ── Serve cache if fresh ──────────────────────────────────────────────────
    with _cache_lock:
        cached_ts   = _cache["ts"]
        cached_data = _cache["data"]

    if cached_ts is not None and cached_data is not None:
        age = now - cached_ts
        if age < CACHE_DURATION_SECONDS:
            result = dict(cached_data)
            result["approved"] = cached_data["approved"] and (
                cached_data["bias"] == "NEUTRAL" or
                cached_data["bias"] == proposed_direction
            )
            result["cached"] = True
            age_m = int(age) // 60
            age_s = int(age) % 60
            if result["approved"]:
                log.info(
                    f"[Regime] serving cached result ({age_m}m {age_s}s old) → APPROVED"
                )
            else:
                log.info(
                    f"[Regime] serving cached result ({age_m}m {age_s}s old) → "
                    f"SKIPPED ({result['reason']})"
                )
            return result

    # ── Fetch + compute ───────────────────────────────────────────────────────
    try:
        candles_15m = fetch_candles("15",  CANDLES_15M)
        candles_4h  = fetch_candles("240", CANDLES_4H)
    except FetchError as exc:
        log.warning(f"[Regime] {exc}")
        return {
            "approved":      False,
            "reason":        "regime data unavailable",
            "regime_15m":    "CHOPPY",
            "regime_4h":     "CHOPPY",
            "adx_15m":       0.0,
            "adx_4h":        0.0,
            "plus_di_15m":   0.0,
            "minus_di_15m":  0.0,
            "plus_di_4h":    0.0,
            "minus_di_4h":   0.0,
            "atr_ratio_15m": 1.0,
            "atr_ratio_4h":  1.0,
            "bbw_ratio_15m": 1.0,
            "bbw_ratio_4h":  1.0,
            "trend_strength": "NONE",
            "bias":          "NEUTRAL",
            "alignment":     False,
            "cached":        False,
        }

    closes_15m = [c["close"] for c in candles_15m]
    closes_4h  = [c["close"] for c in candles_4h]

    adx_15m, pdi_15m, ndi_15m = calc_adx(candles_15m)
    adx_4h,  pdi_4h,  ndi_4h  = calc_adx(candles_4h)
    atr_r_15m = calc_atr_ratio(candles_15m)
    atr_r_4h  = calc_atr_ratio(candles_4h)
    bbw_r_15m = calc_bbw_ratio(candles_15m)
    bbw_r_4h  = calc_bbw_ratio(candles_4h)

    regime_15m = classify_regime(adx_15m, pdi_15m, ndi_15m, atr_r_15m, bbw_r_15m, closes_15m)
    regime_4h  = classify_regime(adx_4h,  pdi_4h,  ndi_4h,  atr_r_4h,  bbw_r_4h,  closes_4h)

    verdict = _evaluate(
        proposed_direction,
        regime_15m, regime_4h,
        adx_15m, adx_4h,
        pdi_15m, ndi_15m,
        pdi_4h,  ndi_4h,
        atr_r_15m, atr_r_4h,
        bbw_r_15m, bbw_r_4h,
    )

    # ── Cache the direction-neutral base result ───────────────────────────────
    with _cache_lock:
        _cache["ts"]   = now
        _cache["data"] = dict(verdict)

    # ── Log verdict ───────────────────────────────────────────────────────────
    _log_verdict(proposed_direction, verdict, adx_15m, pdi_15m, ndi_15m, adx_4h, pdi_4h, ndi_4h)

    return verdict


# ── Logging helpers ───────────────────────────────────────────────────────────

def _log_verdict(
    proposed: str,
    v: dict,
    adx_15m: float, pdi_15m: float, ndi_15m: float,
    adx_4h:  float, pdi_4h:  float, ndi_4h:  float,
) -> None:
    r15 = v["regime_15m"]
    r4h = v["regime_4h"]

    if v["approved"]:
        log.info(
            f"[Regime] 15m={r15}(ADX={adx_15m:.1f}|+DI={pdi_15m:.1f}|-DI={ndi_15m:.1f}) "
            f"4h={r4h}(ADX={adx_4h:.1f}|+DI={pdi_4h:.1f}|-DI={ndi_4h:.1f}) "
            f"proposed={proposed} strength={v['trend_strength']} → APPROVED"
        )
    elif r15 == "CHOPPY" or r4h == "CHOPPY":
        choppy_tf = "15m" if r15 == "CHOPPY" else "4h"
        choppy_adx = adx_15m if r15 == "CHOPPY" else adx_4h
        other_tf   = "4h"   if r15 == "CHOPPY" else "15m"
        other_r    = r4h    if r15 == "CHOPPY" else r15
        other_adx  = adx_4h if r15 == "CHOPPY" else adx_15m
        log.info(
            f"[Regime] {choppy_tf}=CHOPPY(ADX={choppy_adx:.1f}) "
            f"{other_tf}={other_r}(ADX={other_adx:.1f}) "
            f"proposed={proposed} → SKIPPED ({v['reason']})"
        )
    else:
        log.info(
            f"[Regime] 15m={r15}(ADX={adx_15m:.1f}) "
            f"4h={r4h}(ADX={adx_4h:.1f}) "
            f"proposed={proposed} → SKIPPED ({v['reason']})"
        )
