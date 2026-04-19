"""
PolymarketTrader — MAHORAGA integration module
Automatically places FOK bets on Polymarket BTC 15-min markets
whenever the prediction engine fires a high-confidence UP/DOWN signal.
Every bet has a hard 16-minute force-resolve timer.
"""

import os
import sys
import subprocess
import asyncio
import logging
import math
import json
import time
import httpx
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        OrderArgs, OrderType, ApiCreds,
        BalanceAllowanceParams, AssetType,
    )
    _CLOB_AVAILABLE = True
except ImportError:
    ClobClient = None
    OrderArgs = None
    OrderType = None
    ApiCreds = None
    BalanceAllowanceParams = None
    AssetType = None
    _CLOB_AVAILABLE = False

load_dotenv()

log = logging.getLogger("PolymarketTrader")

# ── IGRIS backbone + learner ─────────────────────────────────────────────────
try:
    from igris_backbone import backbone_signal as _backbone_signal_fn
    _BACKBONE_AVAILABLE = True
except ImportError:
    _backbone_signal_fn  = None
    _BACKBONE_AVAILABLE  = False
    log.warning("[IGRIS] igris_backbone.py not found — backbone gate disabled")

try:
    from igris_backbone_fetcher import build_backbone_input, start_odds_polling
    _FETCHER_AVAILABLE = True
except ImportError:
    build_backbone_input = None
    start_odds_polling   = None
    _FETCHER_AVAILABLE   = False

try:
    from igris_learner import IGRISLearner
    _LEARNER_AVAILABLE = True
except ImportError:
    IGRISLearner       = None
    _LEARNER_AVAILABLE = False

CONFIDENCE_THRESHOLD = 85.0
BET_SIZE_USDC        = 1.0
CLOB_HOST            = "https://clob.polymarket.com"
GAMMA_API            = "https://gamma-api.polymarket.com"
CHAIN_ID             = 137


class PolymarketTrader:
    # TRADING RULES — DO NOT CHANGE:
    # 1. ONE bet per 15-min candle — enforced by _bet_candles set (never cleared)
    # 2. NEVER sell or close a position early — hold until market resolves
    # 3. Only exit via: Redeem (win) or $0 redeem (loss) at resolution
    # 4. FOK orders only — no GTC hanging orders
    # 5. ~$1 per bet (shares = round(1 / best_ask, 2)) — no 5-share minimum

    def __init__(self):
        if not _CLOB_AVAILABLE:
            raise ImportError(
                "py-clob-client is not installed. "
                "Run: pip install py-clob-client==0.34.6"
            )

        self.private_key    = os.getenv("POLY_PRIVATE_KEY")
        self.api_key        = os.getenv("POLY_API_KEY")
        self.api_secret     = os.getenv("POLY_SECRET")
        self.api_passphrase = os.getenv("POLY_PASSPHRASE")
        self.funder         = os.getenv("POLY_FUNDER_ADDRESS")

        missing = [name for name, val in {
            "POLY_PRIVATE_KEY":    self.private_key,
            "POLY_API_KEY":        self.api_key,
            "POLY_SECRET":         self.api_secret,
            "POLY_PASSPHRASE":     self.api_passphrase,
            "POLY_FUNDER_ADDRESS": self.funder,
        }.items() if not val]
        if missing:
            raise ValueError(f"Missing Polymarket credentials: {', '.join(missing)}")

        self.client         = None  # initialised in async setup()
        self.active_bets    = []
        self.completed_bets = []
        _base = os.path.dirname(os.path.abspath(__file__))
        self.bets_file           = os.path.join(_base, "polymarket_bets.json")
        self._training_log_file  = os.path.join(_base, "poly_training_log.json")
        self._adaptive_cfg_file  = os.path.join(_base, "poly_adaptive_config.json")
        self._bet_candles         = set()  # tracks every candle that has been bet — never cleared
        self._place_bet_lock      = asyncio.Lock()  # prevents duplicate concurrent bets
        self._wallet_balance      = 0.0
        self._cycle_count         = 0
        self._tor_available       = False
        self._tor_proxies         = {}
        self._use_tor             = False
        # ── Adaptive per-direction thresholds (auto-trained from outcomes) ──
        self._adaptive_thresholds = {"UP": CONFIDENCE_THRESHOLD, "DOWN": CONFIDENCE_THRESHOLD}
        self._resolved_count      = 0  # total resolved bets this session; triggers retrain every 5
        self._last_retrain_info   = {}  # metadata from most recent retrain cycle
        self._load_adaptive_config()
        self._load_bets()  # also rebuilds _bet_candles from history

        # ── Paper trading (simulation mode — no real orders) ─────────────
        self._paper_bets_file   = os.path.join(_base, "polymarket_paper_bets.json")
        self._paper_config_file = os.path.join(_base, "polymarket_paper_config.json")
        self._paper_mode        = False      # enabled/disabled
        self._paper_balance     = 100.0      # current virtual balance
        self._paper_starting    = 100.0      # baseline for P&L calculation
        self._paper_bet_size    = 1.0        # USDC per simulated bet
        self._paper_active      = []         # open paper bets
        self._paper_completed   = []         # settled paper bets (kept all-time)
        self._paper_candles     = set()      # candle lock — separate from real bets
        self._paper_bet_lock    = asyncio.Lock()
        self._load_paper_config()
        self._load_paper_bets()

        # ── IGRIS online learner ──────────────────────────────────────
        self._igris_learner = None
        if _LEARNER_AVAILABLE:
            try:
                self._igris_learner = IGRISLearner()
                log.info("[IGRIS] Online learner initialised")
            except Exception as _ile:
                log.warning(f"[IGRIS] Learner init failed: {_ile}")
        if _FETCHER_AVAILABLE and start_odds_polling is not None:
            try:
                start_odds_polling()
                log.info("[IGRIS] Odds polling thread started")
            except Exception as _ope:
                log.warning(f"[IGRIS] Odds polling start failed: {_ope}")

    # ── Adaptive auto-training ───────────────────────────────────────

    def _load_adaptive_config(self):
        """Load per-direction confidence thresholds learned from past outcomes."""
        try:
            with open(self._adaptive_cfg_file) as f:
                cfg = json.load(f)
                self._adaptive_thresholds["UP"]   = float(cfg.get("UP",   CONFIDENCE_THRESHOLD))
                self._adaptive_thresholds["DOWN"] = float(cfg.get("DOWN", CONFIDENCE_THRESHOLD))
                self._last_retrain_info           = cfg.get("last_retrain", {})
            log.info(
                f"[PolyTrain] Loaded adaptive thresholds — "
                f"UP: {self._adaptive_thresholds['UP']:.1f}%  "
                f"DOWN: {self._adaptive_thresholds['DOWN']:.1f}%"
            )
        except Exception:
            pass  # missing file is fine — defaults are set in __init__

    def _save_adaptive_config(self):
        try:
            cfg = {
                "UP":           self._adaptive_thresholds["UP"],
                "DOWN":         self._adaptive_thresholds["DOWN"],
                "updated_at":   datetime.now(timezone.utc).isoformat(),
                "last_retrain": self._last_retrain_info,
            }
            with open(self._adaptive_cfg_file, "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            log.warning(f"[PolyTrain] Failed to save adaptive config: {e}")

    def _record_outcome(self, bet: dict):
        """
        Append a resolved bet to the training log.
        Called on real AND paper bet resolution — both feed the adaptive thresholds.
        """
        result = bet.get("result")
        if result not in ("won", "lost"):
            return  # don't train on unknowns
        source = "paper" if bet.get("paper") else "real"
        entry = {
            "direction":  bet.get("direction"),
            "confidence": bet.get("confidence"),
            "price_paid": bet.get("price_paid"),
            "result":     result,
            "pnl":        bet.get("pnl", 0),
            "source":     source,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }
        try:
            try:
                with open(self._training_log_file) as f:
                    log_data = json.load(f)
            except Exception:
                log_data = []
            log_data.append(entry)
            # Keep last 200 entries
            if len(log_data) > 200:
                log_data = log_data[-200:]
            with open(self._training_log_file, "w") as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            log.warning(f"[PolyTrain] Failed to write training log: {e}")

        self._resolved_count += 1
        if self._resolved_count % 5 == 0:
            self._retrain()

    def _retrain(self):
        """
        Recompute per-direction confidence thresholds from recent outcomes.
        Called every 5 resolved bets (paper + real combined).
        Results are applied immediately to _adaptive_thresholds — live bets use
        the updated values on the very next candle.

        Threshold adjustment logic (per direction, last 20 bets):
          - Requires ≥ 10 samples before making any change (avoids noise on small sets)
          - accuracy ≥ 55%  → lower threshold by 1 pt  (floor 80%) — real edge confirmed
          - accuracy 45–54% → no change                              — too close to call
          - accuracy < 45%  → raise threshold by 2 pts (ceil 95%)  — losing, tighten up
        """
        try:
            with open(self._training_log_file) as f:
                log_data = json.load(f)
        except Exception:
            return

        now_iso    = datetime.now(timezone.utc).isoformat()
        retrain_meta = {
            "timestamp":   now_iso,
            "total_samples": len(log_data),
            "paper_samples": sum(1 for e in log_data if e.get("source") == "paper"),
            "real_samples":  sum(1 for e in log_data if e.get("source") != "paper"),
            "directions":    {},
        }

        for direction in ("UP", "DOWN"):
            recent   = [e for e in log_data if e.get("direction") == direction][-20:]
            n        = len(recent)
            wins     = sum(1 for e in recent if e["result"] == "won")
            accuracy = wins / n if n else 0.0
            current  = self._adaptive_thresholds[direction]
            paper_n  = sum(1 for e in recent if e.get("source") == "paper")

            if n < 10:
                log.info(
                    f"[PolyTrain] {direction}: only {n} samples — need 10 to retrain "
                    f"({paper_n} paper / {n - paper_n} real)"
                )
                retrain_meta["directions"][direction] = {
                    "samples": n, "wins": wins, "accuracy": round(accuracy, 4),
                    "threshold": current, "action": "skip_low_samples",
                }
                continue

            if accuracy >= 0.55:
                new_thresh = max(80.0, current - 1.0)
                action_key = "lower"
                action_str = f"↓ ({accuracy:.0%} ≥ 55% — real edge)"
            elif accuracy < 0.45:
                new_thresh = min(95.0, current + 2.0)
                action_key = "raise"
                action_str = f"↑ ({accuracy:.0%} < 45% — tighten)"
            else:
                new_thresh = current
                action_key = "hold"
                action_str = f"= ({accuracy:.0%} — borderline, hold)"

            prev = current
            if new_thresh != current:
                self._adaptive_thresholds[direction] = round(new_thresh, 1)

            log.info(
                f"[PolyTrain] {direction}: {prev:.1f}% → {new_thresh:.1f}% {action_str} | "
                f"{wins}/{n} wins | {paper_n}p/{n - paper_n}r samples | "
                f"NOW LIVE for real bets"
            )

            retrain_meta["directions"][direction] = {
                "samples":       n,
                "paper_samples": paper_n,
                "real_samples":  n - paper_n,
                "wins":          wins,
                "accuracy":      round(accuracy, 4),
                "prev_threshold": round(prev, 1),
                "new_threshold":  round(new_thresh, 1),
                "action":        action_key,
            }

        self._last_retrain_info = retrain_meta
        self._save_adaptive_config()

    def get_adaptive_thresholds(self) -> dict:
        return {
            "UP":   self._adaptive_thresholds["UP"],
            "DOWN": self._adaptive_thresholds["DOWN"],
        }

    # ── Tor check ────────────────────────────────────────────────────

    def _check_tor(self) -> bool:
        try:
            result = subprocess.run(["which", "torsocks"], capture_output=True)
            if result.returncode == 0:
                log.info("[PolymarketTrader] Tor available — routing CLOB calls through Tor")
                return True
            log.warning("[PolymarketTrader] torsocks not found — orders may fail due to geo-block")
            return False
        except Exception:
            return False

    # ── Async setup ──────────────────────────────────────────────────

    async def setup(self):
        """Initialise CLOB client, cancel leftover orders, fetch opening balance."""
        self._tor_available = self._check_tor()

        creds = ApiCreds(
            api_key=self.api_key,
            api_secret=self.api_secret,
            api_passphrase=self.api_passphrase,
        )
        self.client = ClobClient(
            host=CLOB_HOST,
            chain_id=CHAIN_ID,
            key=self.private_key,
            creds=creds,
            signature_type=1,
            funder=self.funder,
        )

        self._tor_proxies = {
            "http":  "socks5h://127.0.0.1:9050",
            "https": "socks5h://127.0.0.1:9050",
        }
        self._use_tor = self._tor_available
        log.info("[PolymarketTrader] Tor proxy configured for Polymarket calls only")

        # Cancel any leftover open orders from previous session
        try:
            open_orders = await asyncio.to_thread(self.client.get_orders)
            if open_orders and len(open_orders) > 0:
                await asyncio.to_thread(self.client.cancel_all)
                log.info(f"[PolymarketTrader] Cancelled {len(open_orders)} leftover open orders")
            else:
                log.info("[PolymarketTrader] No leftover open orders")
        except Exception as e:
            log.debug(f"[PolymarketTrader] Order cleanup: {e}")

        await self._refresh_balance()
        log.info(f"[PolymarketTrader] Initialized. Balance: ${self._wallet_balance:.2f} USDC")

    # ── Balance ──────────────────────────────────────────────────────

    async def _refresh_balance(self) -> float:
        """
        Fetch USDC balance from the CLOB.
        Raw value is in micro-USDC (6 decimal places) so divide by 1_000_000.
        Falls back to a direct REST call if the SDK path fails.
        """
        # ── SDK path ──
        if self.client is not None:
            try:
                params = BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,
                    signature_type=1,
                )
                raw = await asyncio.to_thread(self.client.get_balance_allowance, params)
                raw_balance = float(raw.get("balance", "0") or "0")
                self._wallet_balance = round(raw_balance / 1_000_000, 2)
                log.info(f"[PolymarketTrader] Balance: ${self._wallet_balance:.2f} USDC")
                return self._wallet_balance
            except Exception as e:
                log.warning(f"[PolymarketTrader] SDK balance call failed: {e} — trying REST fallback")

        # ── REST fallback via direct CLOB endpoint ──
        try:
            from py_clob_client.headers.headers import create_level_2_headers
            from py_clob_client.clob_types import RequestArgs
            request_args = RequestArgs(method="GET", request_path="/balance-allowance")
            headers = create_level_2_headers(self.client.signer, self.client.creds, request_args)
            url = f"{CLOB_HOST}/balance-allowance?asset_type=COLLATERAL&signature_type=1"
            async with httpx.AsyncClient(timeout=10) as hx:
                resp = await hx.get(url, headers=headers)
                resp.raise_for_status()
                raw = resp.json()
            raw_balance = float(raw.get("balance", "0") or "0")
            self._wallet_balance = round(raw_balance / 1_000_000, 2)
            log.info(f"[PolymarketTrader] Balance (REST): ${self._wallet_balance:.2f} USDC")
        except Exception as e2:
            log.warning(f"[PolymarketTrader] REST balance fallback failed: {e2}")

        return self._wallet_balance

    # ── Market discovery ─────────────────────────────────────────────

    async def find_btc_15min_market(self, direction: str) -> dict | None:
        try:
            import requests
            import json as _json

            # Polymarket BTC 15-min slug format: btc-updown-15m-{candle_open_unix}
            # Try current candle and next 2 candles in case current is almost closed
            now_ts      = int(time.time())
            candle_size = 900  # 15 minutes in seconds

            candidates = []

            for offset in [0, 1, 2]:
                candle_ts = (now_ts // candle_size + offset) * candle_size
                slug      = f"btc-updown-15m-{candle_ts}"

                try:
                    r = requests.get(
                        f"https://gamma-api.polymarket.com/markets?slug={slug}",
                        proxies=self._tor_proxies if self._use_tor else None,
                        timeout=10,
                    )
                    if r.status_code != 200:
                        continue

                    data = r.json()
                    if not data:
                        continue

                    market = data[0]

                    # Hard guard: only accept genuine 15-min BTC markets
                    market_slug = market.get("slug", "")
                    if not market_slug.startswith("btc-updown-15m-"):
                        log.warning(
                            f"[PolyBet] Rejected non-15min market: slug='{market_slug}' — skipping"
                        )
                        continue

                    if market.get("closed") or not market.get("active"):
                        continue

                    end_str = market.get("endDate") or market.get("endDateIso")
                    if not end_str:
                        continue

                    end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))

                    if end_dt <= datetime.now(timezone.utc):
                        continue

                    token_ids = market.get("clobTokenIds", "[]")
                    if isinstance(token_ids, str):
                        token_ids = _json.loads(token_ids)

                    outcomes = market.get("outcomes", '["Up","Down"]')
                    if isinstance(outcomes, str):
                        outcomes = _json.loads(outcomes)

                    target_token = None
                    for i, outcome in enumerate(outcomes):
                        outcome_lower = outcome.lower()
                        if direction == "UP" and outcome_lower in ["up", "yes", "higher", "above"]:
                            target_token = token_ids[i] if i < len(token_ids) else None
                            break
                        if direction == "DOWN" and outcome_lower in ["down", "no", "lower", "below"]:
                            target_token = token_ids[i] if i < len(token_ids) else None
                            break

                    if not target_token:
                        target_token = token_ids[0] if direction == "UP" else token_ids[1]

                    if not target_token:
                        continue

                    prices = market.get("outcomePrices", '["0.5","0.5"]')
                    if isinstance(prices, str):
                        prices = _json.loads(prices)

                    up_price   = float(prices[0]) if len(prices) > 0 else 0.5
                    down_price = float(prices[1]) if len(prices) > 1 else 0.5  # noqa: F841

                    # Skip if market has already nearly resolved — no betting value
                    if up_price > 0.90 or up_price < 0.10:
                        log.info(
                            f"[PolyBet] Skipping {slug} — market already resolved "
                            f"(UP={up_price:.3f})"
                        )
                        continue

                    idx      = 0 if direction == "UP" else 1
                    best_ask = float(prices[idx]) if idx < len(prices) else 0.5

                    minutes_left = int(
                        (end_dt - datetime.now(timezone.utc)).total_seconds() // 60
                    )

                    if market.get("restricted"):
                        log.warning(
                            f"[PolyBet] Market {slug} is restricted — "
                            f"may not be available in your region"
                        )

                    candidates.append({
                        "market_id":    market.get("id"),
                        "token_id":     target_token,
                        "question":     market.get("question", "BTC Up/Down"),
                        "end_time":     end_str,
                        "best_ask":     best_ask,
                        "minutes_left": minutes_left,
                        "slug":         slug,
                    })

                    log.info(
                        f"[PolyBet] Found market: {market.get('question', '')[:55]} "
                        f"| Ends in {minutes_left}m | Odds: {best_ask:.3f} | Slug: {slug}"
                    )

                except Exception as e:
                    log.debug(f"[PolyBet] Slug {slug} failed: {e}")
                    continue

            if not candidates:
                log.warning(f"[PolyBet] No active BTC 15-min market found for {direction}")
                return None

            candidates.sort(key=lambda x: x["minutes_left"], reverse=True)
            chosen = candidates[0]
            log.info(
                f"[PolyBet] Selected: {chosen['question'][:55]} "
                f"| {chosen['minutes_left']}m left | ${chosen['best_ask']:.3f} odds"
            )
            return chosen

        except Exception as e:
            log.error(f"[PolyBet] Market search crashed: {e}")
            return None

    # ── Bet placement ────────────────────────────────────────────────

    async def place_bet(self, direction: str, confidence: float,
                        mahoraga_signal: str | None = None) -> dict | None:
        # Mutex: only one bet can execute at a time — prevents duplicate bets from
        # concurrent callers (autonomous loop + dashboard endpoint both call this)
        if self._place_bet_lock.locked():
            log.debug("[PolyBet] place_bet already in progress — skipping concurrent call")
            return None
        async with self._place_bet_lock:
            return await self._place_bet_impl(direction, confidence, mahoraga_signal)

    async def _place_bet_impl(self, direction: str, confidence: float,
                               mahoraga_signal: str | None = None) -> dict | None:
        try:
            # ── MAHORAGA signal alignment ─────────────────────────────
            # mahoraga_signal is the Bybit autotrader's latest signal: "UP", "DOWN", or None
            if mahoraga_signal and mahoraga_signal not in ("NEUTRAL", "OFFLINE"):
                if mahoraga_signal != direction:
                    log.info(
                        f"[PolyBet] MAHORAGA conflict — MLP says {direction} but "
                        f"MAHORAGA signals {mahoraga_signal} → skip"
                    )
                    return None
                log.info(f"[PolyBet] MAHORAGA confirmed {direction} ✓")

            # ── Adaptive per-direction confidence threshold ───────────
            threshold = self._adaptive_thresholds.get(direction, CONFIDENCE_THRESHOLD)
            if confidence < threshold:
                log.info(
                    f"[PolyBet] Confidence {confidence:.1f}% below adaptive threshold "
                    f"{threshold:.1f}% for {direction} — skip"
                )
                return None

            # HARD candle lock — set of already-bet candles, never cleared or reset
            current_candle = int(time.time()) // 900 * 900
            if current_candle in self._bet_candles:
                log.debug(f"[PolyBet] Candle {current_candle} already bet — hard skip")
                return None

            # Lock immediately before any async work — never removed even on failure
            self._bet_candles.add(current_candle)

            # Safety: cancel stale unfilled orders (NOT filled positions) before placing new one
            try:
                open_orders = await asyncio.to_thread(self.client.get_orders)
                if open_orders:
                    for order in open_orders:
                        order_status = order.get("status", "")
                        if order_status in ("OPEN", "UNMATCHED"):
                            await asyncio.to_thread(self.client.cancel, order.get("id"))
                            log.info(f"[PolyBet] Cancelled stale unfilled order: {order.get('id')}")
            except Exception as _oe:
                log.debug(f"[PolyBet] Order cleanup: {_oe}")

            # Find the market
            market = await self.find_btc_15min_market(direction)
            if not market:
                # Candle remains locked — a failed market search does not grant a retry
                log.warning("[PolyBet] No market found — candle remains locked")
                return None

            token_id  = market["token_id"]
            market_id = market["market_id"]
            question  = market["question"]
            end_time  = market["end_time"]

            # Use price from Gamma API outcomePrices — reliable, no orderbook needed
            best_ask = market["best_ask"]
            minutes_left = market.get("minutes_left", 15)
            log.info(f"[PolyBet] Using Gamma price: {best_ask:.3f} | {minutes_left}m left")

            # ── Filter 1: Market nearly resolved ─────────────────────────
            if best_ask > 0.90 or best_ask < 0.10:
                log.warning(
                    f"[PolyBet] No value — odds {best_ask:.3f} market nearly resolved, skip"
                )
                return None

            # ── Filter 2: Minimum time remaining ────────────────────────
            # Entering with <6 min left gives bad fills and almost no resolution buffer
            if minutes_left < 6:
                log.info(
                    f"[PolyBet] Only {minutes_left}m left in candle — too late to enter, skip"
                )
                return None

            # ── Filter 3: Market edge (avoid pure 50/50 markets) ────────
            # If the market prices the outcome 42¢–58¢, it's priced as a coin-flip.
            # Our MLP must disagree significantly with the market for there to be real edge.
            # We require the market to price it at ≤0.42 or ≥0.58 (crowd has a bias we fade),
            # OR our confidence is extremely high (≥88%) when the market is 50/50.
            market_implied_prob = best_ask  # market price ≈ implied probability
            market_is_coinflip  = 0.42 < market_implied_prob < 0.58
            if market_is_coinflip and confidence < 88.0:
                log.info(
                    f"[PolyBet] Market priced at {best_ask:.3f} (near 50/50) and confidence "
                    f"{confidence:.1f}% < 88% — insufficient edge, skip"
                )
                return None

            # ── Filter 4: Expected Value check ───────────────────────────
            # EV = (our_win_prob * payout) - 1
            # Require at least +10% EV to place a bet
            our_win_prob = confidence / 100.0
            payout       = 1.0 / best_ask          # profit multiplier if we win
            ev           = (our_win_prob * payout) - 1.0
            if ev < 0.10:
                log.info(
                    f"[PolyBet] EV {ev:+.3f} below +0.10 threshold "
                    f"(conf:{confidence:.1f}% odds:{best_ask:.3f}) — skip"
                )
                return None

            log.info(
                f"[PolyBet] Confluence passed — EV:{ev:+.3f} | "
                f"odds:{best_ask:.3f} | conf:{confidence:.1f}% | {minutes_left}m left"
            )

            # ── IGRIS BACKBONE GATE ───────────────────────────────────────────
            # All 4 market signals must agree with direction or bet is skipped.
            # If the backbone extension isn't built, this gate is bypassed.
            _backbone_scores: dict = {}
            _igris_features         = None
            _current_odds: float    = best_ask

            if _BACKBONE_AVAILABLE and _FETCHER_AVAILABLE:
                try:
                    backbone_input = build_backbone_input(direction)
                    if backbone_input is None:
                        # Network/data failure — degrade gracefully, don't block the bet
                        log.info("[IGRIS] Backbone data unavailable — proceeding without backbone gate")
                    else:
                        backbone_result = _backbone_signal_fn(backbone_input)
                        if not backbone_result.get("approved"):
                            reason = backbone_result.get("reason", "")
                            if "conflict" in reason:
                                # Signals actively disagree on direction — skip the bet
                                log.info(f"[IGRIS] Backbone signal conflict: {reason} — skipping bet")
                                return None
                            else:
                                # Inconclusive (insufficient data, empty odds buffer, etc.)
                                # — degrade gracefully, don't block the bet
                                log.info(f"[IGRIS] Backbone inconclusive ({reason}) — proceeding without backbone gate")
                        else:
                            _backbone_scores = backbone_result.get("scores", {})
                            odds_hist     = backbone_input.get("odds_history", [])
                            _current_odds = odds_hist[-1] if odds_hist else best_ask
                            log.info(
                                f"[IGRIS] Backbone approved {direction} ✓ | "
                                f"scores: momentum={_backbone_scores.get('momentum', 0):.4f} "
                                f"ob={_backbone_scores.get('ob_imbalance', 0):.4f} "
                                f"funding={_backbone_scores.get('funding_divergence', 0):.4f} "
                                f"vel={_backbone_scores.get('odds_velocity', 0):.4f}"
                            )
                except Exception as _bg_err:
                    log.warning(f"[IGRIS] Backbone gate error: {_bg_err} — proceeding without it")

            # ── REGIME GATE ───────────────────────────────────────────────────
            try:
                from market_regime import get_regime_verdict as _get_regime
                _rv = _get_regime(direction)
                if not _rv["approved"]:
                    log.info(f"[IGRIS] Bet blocked — {_rv['reason']}")
                    return None
                log.info(
                    f"[IGRIS] Regime cleared — "
                    f"{_rv['regime_15m']} / {_rv['regime_4h']} "
                    f"strength={_rv['trend_strength']} placing bet"
                )
            except Exception as _re:
                log.warning(f"[IGRIS] Regime gate error: {_re} — proceeding")

            # ── IGRIS LEARNER GATE ────────────────────────────────────────────
            # Only active when backbone produced scores (so features are meaningful).
            if self._igris_learner is not None and _backbone_scores:
                try:
                    _igris_features = self._igris_learner.build_features(
                        backbone_scores=_backbone_scores,
                        confidence=confidence / 100.0,
                        odds=_current_odds,
                        direction=direction,
                    )
                    learner_dir, learner_conf = self._igris_learner.predict(_igris_features)
                    if learner_dir == "NONE":
                        log.info(
                            f"[IGRIS] Learner inconclusive (conf={learner_conf:.2f}) — skipping"
                        )
                        return None
                    if learner_dir != direction:
                        log.info(
                            f"[IGRIS] Learner disagrees: upstream={direction} "
                            f"learner={learner_dir} (conf={learner_conf:.2f}) — skipping"
                        )
                        return None
                    log.info(
                        f"[IGRIS] Learner confirmed {direction} "
                        f"(conf={learner_conf:.2f}) ✓"
                    )
                except Exception as _lg_err:
                    log.warning(f"[IGRIS] Learner gate error: {_lg_err} — proceeding without it")
                    _igris_features = None

            # Calculate shares — ~$1 per bet, no artificial minimum
            shares = round(1.0 / best_ask, 2)
            cost   = round(shares * best_ask, 4)

            log.info(
                f"[PolyBet] Placing FOK {direction} | "
                f"{shares} shares @ {best_ask:.3f} = ${cost} | {question[:50]}"
            )

            # Build order script — uses sys.executable and dynamic project path
            # so it works regardless of deployment directory
            _proj  = os.path.dirname(os.path.abspath(__file__))
            _py    = sys.executable
            _price = round(best_ask, 4)

            order_script = (
                f"import sys; sys.path.insert(0, {_proj!r})\n"
                f"from py_clob_client.client import ClobClient\n"
                f"from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType\n"
                f"import json\n"
                f"creds = ApiCreds(api_key={self.api_key!r}, "
                f"api_secret={self.api_secret!r}, "
                f"api_passphrase={self.api_passphrase!r})\n"
                f"client = ClobClient(host='https://clob.polymarket.com', "
                f"chain_id=137, key={self.private_key!r}, creds=creds, "
                f"signature_type=1, funder={self.funder!r})\n"
                f"order = client.create_order(OrderArgs("
                f"token_id={token_id!r}, price={_price}, "
                f"size={shares}, side='BUY'))\n"
                f"resp = client.post_order(order, OrderType.FOK)\n"
                f"print(json.dumps(resp) if isinstance(resp, dict) else str(resp))\n"
            )

            cmd = ["torsocks", _py, "-c", order_script] if self._use_tor else [_py, "-c", order_script]
            result = await asyncio.to_thread(
                subprocess.run, cmd,
                capture_output=True, text=True, timeout=30,
            )

            if result.returncode != 0:
                log.error(f"[PolyBet] Order subprocess failed: {result.stderr[:200]}")
                # Retry once with slightly higher price to improve fill chance
                _price_retry  = min(0.95, round(best_ask + 0.02, 2))
                _shares_retry = round(1.0 / _price_retry, 2)
                order_script_retry = order_script \
                    .replace(f"price={_price}", f"price={_price_retry}") \
                    .replace(f"size={shares}", f"size={_shares_retry}")
                cmd_retry = ["torsocks", _py, "-c", order_script_retry] if self._use_tor else [_py, "-c", order_script_retry]
                result = await asyncio.to_thread(
                    subprocess.run, cmd_retry,
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode != 0:
                    log.error(f"[PolyBet] Retry also failed: {result.stderr[:200]}")
                    return None

            # Parse response
            try:
                response = json.loads(result.stdout.strip())
            except Exception:
                response = {"raw": result.stdout.strip()}

            log.info(f"[PolyBet] Order response: {response}")

            # Geo-restriction check
            if isinstance(response, dict):
                err_msg = str(
                    response.get("error", "")
                    or response.get("message", "")
                    or response.get("detail", "")
                ).lower()
                if any(kw in err_msg for kw in ("restricted", "geo", "blocked", "not available")):
                    log.error(
                        "[PolyBet] Order rejected — geo-restriction. "
                        "Server needs VPN to a supported country."
                    )
                    return None

            # Check if FOK was cancelled (no liquidity at that price)
            status = response.get("status", "") if isinstance(response, dict) else ""
            if status == "cancelled":
                log.warning("[PolyBet] FOK cancelled — no liquidity at price")
                return None

            # Build bet record
            order_id = (
                response.get("orderID")
                or response.get("order_id")
                or response.get("id")
                or f"local_{current_candle}"
            ) if isinstance(response, dict) else f"local_{current_candle}"

            bet = {
                "id":               str(order_id),
                "direction":        direction,
                "confidence":       round(confidence, 2),
                "market_id":        market_id,
                "token_id":         token_id,
                "question":         question,
                "price_paid":       best_ask,
                "shares":           shares,
                "bet_usdc":         BET_SIZE_USDC,
                "bet_cost":         cost,
                "potential_profit": round(shares - cost, 4),
                "placed_at":        datetime.now(timezone.utc).isoformat(),
                "end_time":         end_time,
                "status":           "open",
                "result":           None,
                "pnl":              None,
                # IGRIS learner features — stored so we can update after resolution
                "igris_features":   _igris_features.flatten().tolist() if _igris_features is not None else None,
            }

            self.active_bets.append(bet)
            self._save_bets()

            log.info(
                f"[PolyBet] ✓ PLACED ${cost} on {direction} | "
                f"Market: {question[:50]} | "
                f"Odds: {best_ask:.3f} | "
                f"Potential profit: ${bet['potential_profit']}"
            )

            # Schedule forced resolution check after 16 minutes
            # (15-min candle + 1-min buffer for Chainlink resolution)
            asyncio.create_task(self._force_resolve_after(bet, delay_seconds=960))

            await self._refresh_balance()
            return bet

        except Exception as e:
            log.error(f"[PolyBet] _place_bet_impl crashed: {e}")
            return None

    # ── Force resolve (ENFORCER) ─────────────────────────────────────

    async def _force_resolve_after(self, bet: dict, delay_seconds: int = 960):
        """
        Fires 16 minutes after bet placement.
        Forces resolution check and claim attempt.
        Retries every 120s for up to 20 attempts (40 min).
        Total window: ~56 min — covers Polymarket's slow 30-45 min closure delay.
        """
        await asyncio.sleep(delay_seconds)

        log.info(f"[PolyBet] ⏰ Force resolve triggered for: {bet['question'][:50]}")

        # Skip if already resolved by the check_and_claim() sweep
        if any(b["id"] == bet["id"] for b in self.completed_bets):
            log.info("[PolyBet] Already resolved by check_and_claim — skipping force resolve")
            return

        import requests as _req

        for attempt in range(20):
            try:
                r = _req.get(
                    f"{GAMMA_API}/markets/{bet['market_id']}",
                    proxies=self._tor_proxies if self._use_tor else None,
                    timeout=10,
                )
                market_data = r.json()

                resolved = market_data.get("resolved", False)
                closed   = market_data.get("closed", False)

                if resolved or closed:
                    winning_outcome = (
                        market_data.get("resolvedOutcome")
                        or market_data.get("winner")
                        or market_data.get("resolutionOutcome")
                        or ""
                    ).lower()

                    # Fallback: infer from outcomePrices when explicit winner field absent
                    # Polymarket often sets closed=True but winner=None — outcomePrices is reliable
                    if not winning_outcome:
                        prices = market_data.get("outcomePrices", '["0.5","0.5"]')
                        if isinstance(prices, str):
                            import json as _j
                            prices = _j.loads(prices)
                        up_price = float(prices[0]) if prices else 0.5
                        if up_price > 0.85:
                            winning_outcome = "up"
                            log.info(f"[PolyBet] Inferred winner=UP from outcomePrices ({up_price:.3f})")
                        elif up_price < 0.15:
                            winning_outcome = "down"
                            log.info(f"[PolyBet] Inferred winner=DOWN from outcomePrices ({up_price:.3f})")

                    up_kw   = ["up", "yes", "higher", "above"]
                    down_kw = ["down", "no", "lower", "below"]

                    won = False
                    if bet["direction"] == "UP":
                        won = any(kw in winning_outcome for kw in up_kw)
                    elif bet["direction"] == "DOWN":
                        won = any(kw in winning_outcome for kw in down_kw)

                    # If winning_outcome still indeterminate, wait for cleaner data
                    if not winning_outcome:
                        log.info(
                            f"[PolyBet] Market closed but winner indeterminate "
                            f"(attempt {attempt + 1}/20) — retry in 120s"
                        )
                        await asyncio.sleep(120)
                        continue

                    if won:
                        pnl = round(bet["shares"] - bet["bet_cost"], 4)
                        bet.update({"result": "won", "pnl": pnl, "status": "completed"})
                        await self._claim_winnings(bet)
                        log.info(
                            f"[PolyBet] ✓ WON +${pnl:.4f} | "
                            f"{bet['question'][:50]} | "
                            f"Balance: ${self._wallet_balance:.2f}"
                        )
                    else:
                        pnl = -round(bet["bet_cost"], 4)
                        bet.update({"result": "lost", "pnl": pnl, "status": "completed"})
                        log.info(
                            f"[PolyBet] ✗ LOST ${abs(pnl):.4f} | "
                            f"{bet['question'][:50]} | "
                            f"Balance: ${self._wallet_balance:.2f}"
                        )

                    self._record_outcome(bet)

                    # IGRIS online learning update
                    if self._igris_learner is not None:
                        _feats = bet.get("igris_features")
                        if _feats is not None:
                            try:
                                import numpy as _np
                                self._igris_learner.update(
                                    features=_np.array(_feats, dtype=_np.float64).reshape(1, -1),
                                    actual_outcome="WON" if won else "LOST",
                                )
                            except Exception as _ue:
                                log.debug(f"[IGRIS] Learner update error: {_ue}")

                    if bet in self.active_bets:
                        self.active_bets.remove(bet)
                    self.completed_bets.append(bet)
                    self._save_bets()
                    await self._refresh_balance()
                    return

                else:
                    log.info(
                        f"[PolyBet] Market not yet closed "
                        f"(attempt {attempt + 1}/20) — retry in 120s"
                    )
                    await asyncio.sleep(120)

            except Exception as e:
                log.error(f"[PolyBet] Force resolve attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(120)

        # 20 attempts exhausted (~56 min total) — mark as unknown
        log.warning(
            f"[PolyBet] ⚠ Could not resolve after 20 attempts (~56min): "
            f"{bet['question'][:50]} — marking as unknown"
        )
        bet.update({"result": "unknown", "status": "completed", "pnl": 0})
        if bet in self.active_bets:
            self.active_bets.remove(bet)
        self.completed_bets.append(bet)
        self._save_bets()

    # ── Claim winnings ───────────────────────────────────────────────

    async def _claim_winnings(self, bet: dict):
        """
        Attempts to redeem winning position via subprocess.
        Called automatically after a win is detected by _force_resolve_after().
        Polymarket often auto-settles, so failure here is non-critical.
        """
        try:
            _proj = os.path.dirname(os.path.abspath(__file__))
            _py   = sys.executable

            claim_script = (
                f"import sys; sys.path.insert(0, {_proj!r})\n"
                f"from py_clob_client.client import ClobClient\n"
                f"from py_clob_client.clob_types import ApiCreds\n"
                f"import json\n"
                f"creds = ApiCreds(api_key={self.api_key!r}, "
                f"api_secret={self.api_secret!r}, "
                f"api_passphrase={self.api_passphrase!r})\n"
                f"client = ClobClient(host='https://clob.polymarket.com', "
                f"chain_id=137, key={self.private_key!r}, creds=creds, "
                f"signature_type=1, funder={self.funder!r})\n"
                f"try:\n"
                f"    from py_clob_client.clob_types import RedeemPositionsParams\n"
                f"    result = client.redeem_positions(\n"
                f"        RedeemPositionsParams(condition_id={bet.get('market_id', '')!r})\n"
                f"    )\n"
                f"    print(json.dumps({{'status': 'claimed', 'result': str(result)}}))\n"
                f"except Exception as e:\n"
                f"    print(json.dumps({{'status': 'claim_failed', 'error': str(e)}}))\n"
            )

            cmd = ["torsocks", _py, "-c", claim_script] if self._use_tor else [_py, "-c", claim_script]
            result = await asyncio.to_thread(
                subprocess.run, cmd,
                capture_output=True, text=True, timeout=30,
            )

            if result.returncode == 0:
                resp = json.loads(result.stdout.strip())
                if resp.get("status") == "claimed":
                    log.info(f"[PolyBet] 💰 Auto-claimed winnings for {bet['question'][:40]}")
                else:
                    log.warning(f"[PolyBet] Claim attempt: {resp.get('error', 'unknown')}")
            else:
                log.warning(f"[PolyBet] Claim subprocess failed: {result.stderr[:100]}")

        except Exception as e:
            log.warning(f"[PolyBet] _claim_winnings failed: {e} — manual claim may be needed")

    # ── Resolution sweep (safety net) ───────────────────────────────

    async def check_and_claim(self):
        """
        Background sweep — runs every 60s via run_loop().
        Catches anything _force_resolve_after() missed (e.g. server restarts
        where the asyncio task was lost but the bet was persisted to disk).

        Resolution logic:
          1. Not past end_time → keep active
          2. Past end_time, market resolved/closed → determine outcome, move to completed
          3. Past end_time + 30 min, still not resolved → force-expire as unknown
          4. Recently ended but unresolved → keep active, retry next cycle
        """
        if not self.active_bets:
            return

        now          = datetime.now(timezone.utc)
        still_active = []

        for bet in list(self.active_bets):
            try:
                end_dt = datetime.fromisoformat(
                    bet["end_time"].replace("Z", "+00:00")
                )
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)

                # Not yet expired — keep waiting
                if now < end_dt:
                    still_active.append(bet)
                    continue

                seconds_past = (now - end_dt).total_seconds()

                # Query market status
                market_data = {}
                try:
                    async with httpx.AsyncClient(timeout=10) as hx:
                        resp = await hx.get(f"{GAMMA_API}/markets/{bet['market_id']}")
                    if resp.status_code == 200:
                        market_data = resp.json()
                except Exception as fetch_err:
                    log.debug(f"[PolyBet] check_and_claim fetch failed: {fetch_err}")

                resolved = market_data.get("resolved", False)
                closed   = market_data.get("closed", False)

                if resolved or closed:
                    # ── Determine winner ──────────────────────────────
                    resolved_outcome = str(
                        market_data.get("winner")
                        or market_data.get("resolvedOutcome")
                        or market_data.get("resolutionOutcome")
                        or market_data.get("outcome", "")
                    ).lower()

                    # Fallback: infer from outcomePrices when explicit field absent
                    if not resolved_outcome:
                        prices = market_data.get("outcomePrices", '["0.5","0.5"]')
                        if isinstance(prices, str):
                            prices = json.loads(prices)
                        up_price = float(prices[0]) if prices else 0.5
                        if up_price > 0.85:
                            resolved_outcome = "up"
                        elif up_price < 0.15:
                            resolved_outcome = "down"

                    direction = bet["direction"]
                    won = (
                        direction == "UP"
                        and any(kw in resolved_outcome for kw in ("up", "yes", "higher", "above"))
                    ) or (
                        direction == "DOWN"
                        and any(kw in resolved_outcome for kw in ("down", "no", "lower", "below"))
                    )

                    if won:
                        # Attempt auto-claim (non-critical — Polymarket often auto-settles)
                        try:
                            if self.client and hasattr(self.client, "redeem_positions"):
                                await asyncio.to_thread(
                                    self.client.redeem_positions, bet["token_id"]
                                )
                        except Exception as ce:
                            log.debug(f"[PolyBet] Redemption attempt (may auto-settle): {ce}")

                        actual_pnl = round(
                            bet["shares"] - bet.get("bet_cost", bet.get("bet_usdc", 1.0)), 4
                        )
                        bet.update({"status": "completed", "result": "won", "pnl": actual_pnl})
                        log.info(f"[PolyBet] ✓ WON +${actual_pnl:.4f} | {bet['question'][:60]}")
                    else:
                        loss = -round(bet.get("bet_cost", bet.get("bet_usdc", 1.0)), 4)
                        bet.update({"status": "completed", "result": "lost", "pnl": loss})
                        log.info(f"[PolyBet] ✗ LOST ${abs(loss):.4f} | {bet['question'][:60]}")

                    self._record_outcome(bet)

                    # IGRIS online learning update
                    if self._igris_learner is not None:
                        _feats = bet.get("igris_features")
                        if _feats is not None:
                            try:
                                import numpy as _np
                                self._igris_learner.update(
                                    features=_np.array(_feats, dtype=_np.float64).reshape(1, -1),
                                    actual_outcome="WON" if won else "LOST",
                                )
                            except Exception as _ue:
                                log.debug(f"[IGRIS] Learner update error: {_ue}")

                    self.completed_bets.append(bet)
                    self._save_bets()
                    await self._refresh_balance()

                elif seconds_past > 3600:
                    # 60 minutes past end_time, still unresolved — force-expire
                    # (Polymarket can take 30-45 min to flip closed=true after end_time)
                    log.warning(
                        f"[PolyBet] ⚠ Bet {bet['id'][:20]}… unresolved 60min+ past end_time "
                        f"— force-expiring as unknown"
                    )
                    bet.update({"status": "completed", "result": "unknown", "pnl": 0})
                    self.completed_bets.append(bet)
                    self._save_bets()

                else:
                    # Recently ended, resolution pending — retry next cycle
                    log.debug(
                        f"[PolyBet] Bet past end_time by {int(seconds_past)}s, "
                        f"awaiting Polymarket resolution…"
                    )
                    still_active.append(bet)

            except Exception as e:
                log.warning(f"[PolyBet] check_and_claim error for {bet.get('id')}: {e}")
                still_active.append(bet)

        self.active_bets = still_active

    # ── Persistence ──────────────────────────────────────────────────

    def _load_bets(self):
        try:
            with open(self.bets_file) as f:
                data = json.load(f)
                self.active_bets    = data.get("active", [])
                self.completed_bets = data.get("completed", [])
        except Exception:
            pass

        # Rebuild candle lock from persisted bets so restarts never double-bet a candle
        self._bet_candles = set()
        for bet in self.active_bets + self.completed_bets:
            placed_at = bet.get("placed_at")
            if placed_at:
                try:
                    dt = datetime.fromisoformat(placed_at.replace("Z", "+00:00"))
                    candle_ts = int(dt.timestamp()) // 900 * 900
                    self._bet_candles.add(candle_ts)
                except Exception:
                    pass
        if self._bet_candles:
            log.info(f"[PolyBet] Restored {len(self._bet_candles)} candle locks from history")

    def _save_bets(self):
        try:
            with open(self.bets_file, "w") as f:
                json.dump(
                    {"active": self.active_bets, "completed": self.completed_bets},
                    f, indent=2,
                )
        except Exception as e:
            log.warning(f"[PolymarketTrader] Failed to save bets: {e}")

    # ── Paper trading persistence ────────────────────────────────────

    def _load_paper_config(self):
        try:
            with open(self._paper_config_file) as f:
                cfg = json.load(f)
            self._paper_mode     = bool(cfg.get("enabled", False))
            self._paper_balance  = float(cfg.get("balance", 100.0))
            self._paper_starting = float(cfg.get("starting_balance", 100.0))
            self._paper_bet_size = float(cfg.get("bet_size", 1.0))
            log.info(
                f"[PaperBet] Config loaded — mode={'ON' if self._paper_mode else 'OFF'} "
                f"balance=${self._paper_balance:.2f} bet_size=${self._paper_bet_size:.2f}"
            )
        except Exception:
            pass  # missing file is fine — defaults set in __init__

    def _save_paper_config(self):
        try:
            with open(self._paper_config_file, "w") as f:
                json.dump({
                    "enabled":          self._paper_mode,
                    "balance":          round(self._paper_balance, 6),
                    "starting_balance": round(self._paper_starting, 6),
                    "bet_size":         round(self._paper_bet_size, 2),
                    "updated_at":       datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
        except Exception as e:
            log.warning(f"[PaperBet] Failed to save config: {e}")

    def _load_paper_bets(self):
        try:
            with open(self._paper_bets_file) as f:
                data = json.load(f)
            self._paper_active    = data.get("active", [])
            self._paper_completed = data.get("completed", [])
        except Exception:
            return
        # Rebuild candle lock from history
        self._paper_candles = set()
        for bet in self._paper_active + self._paper_completed:
            placed_at = bet.get("placed_at")
            if placed_at:
                try:
                    dt = datetime.fromisoformat(placed_at.replace("Z", "+00:00"))
                    self._paper_candles.add(int(dt.timestamp()) // 900 * 900)
                except Exception:
                    pass
        if self._paper_active:
            log.info(f"[PaperBet] Restored {len(self._paper_active)} open paper bets from disk")

    def _save_paper_bets(self):
        try:
            with open(self._paper_bets_file, "w") as f:
                json.dump({
                    "active":    self._paper_active,
                    "completed": self._paper_completed,
                }, f, indent=2)
        except Exception as e:
            log.warning(f"[PaperBet] Failed to save bets: {e}")

    # ── Paper bet placement ──────────────────────────────────────────

    async def place_paper_bet(self, direction: str, confidence: float) -> dict | None:
        """
        Simulate a Polymarket bet — identical filters to real bets but zero real execution.
        Fires from the autonomous loop whenever paper mode is enabled, independently
        of whether real trading is on/off.
        """
        if not self._paper_mode:
            return None
        if self._paper_bet_lock.locked():
            return None
        async with self._paper_bet_lock:
            return await self._place_paper_bet_impl(direction, confidence)

    async def _place_paper_bet_impl(self, direction: str, confidence: float) -> dict | None:
        try:
            # ── Candle lock (separate from real bets) ──────────────────
            current_candle = int(time.time()) // 900 * 900
            if current_candle in self._paper_candles:
                log.debug(f"[PaperBet] Candle {current_candle} already paper-bet — skip")
                return None
            self._paper_candles.add(current_candle)

            # ── Adaptive confidence threshold ───────────────────────────
            threshold = self._adaptive_thresholds.get(direction, CONFIDENCE_THRESHOLD)
            if confidence < threshold:
                log.info(
                    f"[PaperBet] Conf {confidence:.1f}% < threshold {threshold:.1f}% "
                    f"for {direction} — skip"
                )
                return None

            # ── Find market (same as real bets) ────────────────────────
            market = await self.find_btc_15min_market(direction)
            if not market:
                log.info("[PaperBet] No market found — skip")
                return None

            best_ask     = market["best_ask"]
            minutes_left = market.get("minutes_left", 15)

            # ── Same filters as real bets ──────────────────────────────
            if best_ask > 0.90 or best_ask < 0.10:
                log.info(f"[PaperBet] Market nearly resolved ({best_ask:.3f}) — skip")
                return None
            if minutes_left < 6:
                log.info(f"[PaperBet] Only {minutes_left}m left — skip")
                return None
            if 0.42 < best_ask < 0.58 and confidence < 88.0:
                log.info(
                    f"[PaperBet] Near 50/50 ({best_ask:.3f}) and conf {confidence:.1f}% < 88% — skip"
                )
                return None
            ev = (confidence / 100.0 * (1.0 / best_ask)) - 1.0
            if ev < 0.10:
                log.info(f"[PaperBet] EV {ev:+.3f} below +0.10 — skip")
                return None

            # ── REGIME GATE ────────────────────────────────────────────
            try:
                from market_regime import get_regime_verdict as _get_regime
                _rv = _get_regime(direction)
                if not _rv["approved"]:
                    log.info(f"[PaperBet] Bet blocked — {_rv['reason']}")
                    return None
                log.info(
                    f"[PaperBet] Regime cleared — "
                    f"{_rv['regime_15m']} / {_rv['regime_4h']} "
                    f"strength={_rv['trend_strength']} placing paper bet"
                )
            except Exception as _re:
                log.warning(f"[PaperBet] Regime gate error: {_re} — proceeding")

            # ── Sizing ─────────────────────────────────────────────────
            shares = round(self._paper_bet_size / best_ask, 2)
            cost   = round(shares * best_ask, 4)

            if cost > self._paper_balance:
                log.info(
                    f"[PaperBet] Insufficient paper balance "
                    f"(${self._paper_balance:.2f} < ${cost:.4f}) — skip"
                )
                return None

            # Deduct cost upfront (returned on win as full share value)
            self._paper_balance = round(self._paper_balance - cost, 6)

            bet = {
                "id":               f"paper_{current_candle}",
                "direction":        direction,
                "confidence":       round(confidence, 2),
                "market_id":        market["market_id"],
                "token_id":         market["token_id"],
                "question":         market["question"],
                "price_paid":       best_ask,
                "shares":           shares,
                "bet_usdc":         self._paper_bet_size,
                "bet_cost":         cost,
                "potential_profit": round(shares - cost, 6),
                "ev":               round(ev, 4),
                "placed_at":        datetime.now(timezone.utc).isoformat(),
                "end_time":         market["end_time"],
                "status":           "open",
                "result":           None,
                "pnl":              None,
                "paper":            True,
            }

            self._paper_active.append(bet)
            self._save_paper_bets()
            self._save_paper_config()

            log.info(
                f"[PaperBet] 📄 SIMULATED {direction} | "
                f"{shares} shares @ {best_ask:.3f} = ${cost:.4f} | "
                f"EV:{ev:+.3f} | {minutes_left}m left | "
                f"Paper balance: ${self._paper_balance:.4f}"
            )

            asyncio.create_task(self._paper_resolve_after(bet, delay_seconds=960))
            return bet

        except Exception as e:
            log.error(f"[PaperBet] _place_paper_bet_impl crashed: {e}")
            return None

    async def _paper_resolve_after(self, bet: dict, delay_seconds: int = 960):
        """
        Poll Gamma API until market resolves, then record win/loss and update balance.
        Mirrors _force_resolve_after but updates paper state only.
        """
        await asyncio.sleep(delay_seconds)

        # Skip if already settled (e.g. by _check_paper_bets sweep after restart)
        if any(b["id"] == bet["id"] for b in self._paper_completed):
            return

        log.info(f"[PaperBet] ⏰ Force resolve: {bet['question'][:50]}")
        import requests as _req

        for attempt in range(20):
            try:
                # Guard: check every attempt in case _check_paper_bets already settled this bet
                if any(b["id"] == bet["id"] for b in self._paper_completed):
                    log.debug(f"[PaperBet] {bet['id']} already settled by sweep — skip resolve task")
                    return

                r = _req.get(
                    f"{GAMMA_API}/markets/{bet['market_id']}",
                    proxies=self._tor_proxies if self._use_tor else None,
                    timeout=10,
                )
                mdata = r.json()
                resolved = mdata.get("resolved", False)
                closed   = mdata.get("closed", False)

                if resolved or closed:
                    winning_outcome = (
                        mdata.get("resolvedOutcome")
                        or mdata.get("winner")
                        or mdata.get("resolutionOutcome")
                        or ""
                    ).lower()

                    if not winning_outcome:
                        prices = mdata.get("outcomePrices", '["0.5","0.5"]')
                        if isinstance(prices, str):
                            import json as _j
                            prices = _j.loads(prices)
                        up_price = float(prices[0]) if prices else 0.5
                        if up_price > 0.85:
                            winning_outcome = "up"
                        elif up_price < 0.15:
                            winning_outcome = "down"

                    if not winning_outcome:
                        await asyncio.sleep(120)
                        continue

                    up_kw   = ["up", "yes", "higher", "above"]
                    down_kw = ["down", "no", "lower", "below"]
                    won = (
                        (bet["direction"] == "UP"   and any(kw in winning_outcome for kw in up_kw))
                        or (bet["direction"] == "DOWN" and any(kw in winning_outcome for kw in down_kw))
                    )

                    if won:
                        pnl = round(bet["shares"] - bet["bet_cost"], 6)
                        self._paper_balance = round(self._paper_balance + bet["shares"], 6)
                        bet.update({"result": "won", "pnl": pnl, "status": "completed"})
                        log.info(
                            f"[PaperBet] ✓ WON +${pnl:.4f} | "
                            f"Paper balance: ${self._paper_balance:.4f}"
                        )
                    else:
                        pnl = -round(bet["bet_cost"], 6)
                        bet.update({"result": "lost", "pnl": pnl, "status": "completed"})
                        log.info(
                            f"[PaperBet] ✗ LOST ${abs(pnl):.4f} | "
                            f"Paper balance: ${self._paper_balance:.4f}"
                        )

                    # Feed outcome into adaptive training — paper results adjust real thresholds
                    self._record_outcome(bet)
                    log.info(
                        f"[PaperBet] → Fed {bet['result'].upper()} into adaptive trainer "
                        f"(UP:{self._adaptive_thresholds['UP']:.1f}% "
                        f"DOWN:{self._adaptive_thresholds['DOWN']:.1f}%)"
                    )

                    if bet in self._paper_active:
                        self._paper_active.remove(bet)
                    self._paper_completed.append(bet)
                    self._save_paper_bets()
                    self._save_paper_config()
                    return

                else:
                    await asyncio.sleep(120)

            except Exception as e:
                log.error(f"[PaperBet] Resolve attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(120)

        # 20 attempts exhausted — mark unknown (cost already deducted, no refund)
        bet.update({"result": "unknown", "status": "completed", "pnl": 0})
        if bet in self._paper_active:
            self._paper_active.remove(bet)
        self._paper_completed.append(bet)
        self._save_paper_bets()

    async def _check_paper_bets(self):
        """
        Background sweep for paper bets that survived a restart
        (their asyncio tasks were lost, but bets were persisted to disk).
        Mirrors check_and_claim() for real bets.
        """
        if not self._paper_active:
            return

        now          = datetime.now(timezone.utc)
        still_active = []
        import requests as _req

        for bet in list(self._paper_active):
            try:
                # Guard: skip if already in completed (race with _paper_resolve_after task)
                if any(b["id"] == bet["id"] for b in self._paper_completed):
                    log.debug(f"[PaperBet] sweep: {bet['id']} already in completed — skip")
                    continue

                end_dt = datetime.fromisoformat(bet["end_time"].replace("Z", "+00:00"))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)

                if now < end_dt:
                    still_active.append(bet)
                    continue

                seconds_past = (now - end_dt).total_seconds()

                try:
                    r = _req.get(
                        f"{GAMMA_API}/markets/{bet['market_id']}",
                        proxies=self._tor_proxies if self._use_tor else None,
                        timeout=10,
                    )
                    mdata = r.json() if r.status_code == 200 else {}
                except Exception:
                    mdata = {}

                resolved = mdata.get("resolved", False) or mdata.get("closed", False)
                if resolved:
                    winning_outcome = (
                        mdata.get("resolvedOutcome")
                        or mdata.get("winner")
                        or mdata.get("resolutionOutcome")
                        or ""
                    ).lower()

                    if not winning_outcome:
                        prices = mdata.get("outcomePrices", '["0.5","0.5"]')
                        if isinstance(prices, str):
                            import json as _j
                            prices = _j.loads(prices)
                        up_price = float(prices[0]) if prices else 0.5
                        if up_price > 0.85:
                            winning_outcome = "up"
                        elif up_price < 0.15:
                            winning_outcome = "down"

                    if winning_outcome:
                        won = (
                            (bet["direction"] == "UP"   and any(kw in winning_outcome for kw in ["up","yes","higher","above"]))
                            or (bet["direction"] == "DOWN" and any(kw in winning_outcome for kw in ["down","no","lower","below"]))
                        )
                        if won:
                            pnl = round(bet["shares"] - bet["bet_cost"], 6)
                            self._paper_balance = round(self._paper_balance + bet["shares"], 6)
                            bet.update({"result": "won", "pnl": pnl, "status": "completed"})
                        else:
                            pnl = -round(bet["bet_cost"], 6)
                            bet.update({"result": "lost", "pnl": pnl, "status": "completed"})
                        # Feed into adaptive training
                        self._record_outcome(bet)
                        self._paper_completed.append(bet)
                        self._save_paper_bets()
                        self._save_paper_config()
                        continue

                elif seconds_past > 3600:
                    bet.update({"result": "unknown", "status": "completed", "pnl": 0})
                    self._paper_completed.append(bet)
                    self._save_paper_bets()
                    continue
                else:
                    still_active.append(bet)

            except Exception as e:
                log.warning(f"[PaperBet] sweep error for {bet.get('id')}: {e}")
                still_active.append(bet)

        self._paper_active = still_active

    def get_paper_status(self) -> dict:
        total_won     = len([b for b in self._paper_completed if b.get("result") == "won"])
        total_lost    = len([b for b in self._paper_completed if b.get("result") == "lost"])
        total_settled = total_won + total_lost
        total_pnl     = sum(b["pnl"] for b in self._paper_completed if b.get("pnl") is not None)
        win_rate      = round(total_won / total_settled * 100, 1) if total_settled else 0.0
        current_candle = int(time.time()) // 900 * 900
        return {
            "enabled":          self._paper_mode,
            "balance":          round(self._paper_balance, 4),
            "starting_balance": round(self._paper_starting, 4),
            "bet_size":         round(self._paper_bet_size, 2),
            "pnl":              round(total_pnl, 4),
            "total_bets":       len(self._paper_active) + len(self._paper_completed),
            "total_won":        total_won,
            "total_lost":       total_lost,
            "win_rate":         win_rate,
            "active_bets":      self._paper_active,
            "recent_bets":      list(reversed(self._paper_completed))[:10],
            "candle_locked":    current_candle in self._paper_candles,
        }

    # ── Status ───────────────────────────────────────────────────────

    def get_status(self) -> dict:
        total_won     = len([b for b in self.completed_bets if b.get("result") == "won"])
        total_lost    = len([b for b in self.completed_bets if b.get("result") == "lost"])
        total_settled = total_won + total_lost
        total_pnl     = sum(b["pnl"] for b in self.completed_bets if b.get("pnl") is not None)
        win_rate      = round(total_won / total_settled * 100, 1) if total_settled > 0 else 0.0
        current_candle = int(time.time()) // 900 * 900
        return {
            "balance_usdc":          round(self._wallet_balance, 2),
            "active_bets":           self.active_bets,
            "completed_bets":        self.completed_bets[-20:],
            "total_bets":            len(self.active_bets) + len(self.completed_bets),
            "total_won":             total_won,
            "total_lost":            total_lost,
            "total_pnl":             round(total_pnl, 4),
            "win_rate":              win_rate,
            "bet_candles_count":     len(self._bet_candles),
            "current_candle_locked": current_candle in self._bet_candles,
            "adaptive_threshold_up":   round(self._adaptive_thresholds["UP"],   1),
            "adaptive_threshold_down": round(self._adaptive_thresholds["DOWN"], 1),
            "training": {
                "resolved_count":  self._resolved_count,
                "last_retrain":    self._last_retrain_info,
                "next_retrain_in": 5 - (self._resolved_count % 5) if self._resolved_count % 5 != 0 else 5,
            },
        }

    # ── Background run loop ──────────────────────────────────────────

    async def run_loop(self):
        """
        Runs every 60s. check_and_claim() is the safety net for bets that
        survived a server restart (asyncio tasks don't persist across restarts).
        Balance is logged every 5 cycles.
        """
        cycle = 0
        while True:
            try:
                await asyncio.sleep(60)
                cycle += 1

                # Always sweep for resolved bets (real + paper)
                await self.check_and_claim()
                await self._check_paper_bets()

                # Every 5 cycles: refresh balance and log full status
                if cycle % 5 == 0:
                    await self._refresh_balance()
                    settled_pnl = sum(
                        b["pnl"] for b in self.completed_bets if b.get("pnl") is not None
                    )
                    log.info(
                        f"[PolymarketTrader] Status — "
                        f"Balance: ${self._wallet_balance:.2f} | "
                        f"Active bets: {len(self.active_bets)} | "
                        f"Completed: {len(self.completed_bets)} | "
                        f"Total PnL: ${settled_pnl:.4f}"
                    )

            except Exception as e:
                log.error(f"[PolymarketTrader] run_loop error: {e}")
                await asyncio.sleep(10)
