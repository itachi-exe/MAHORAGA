"""
PolymarketTrader — MAHORAGA integration module
Automatically places bets on Polymarket BTC 15-min markets
whenever the prediction engine fires a high-confidence UP/DOWN signal.
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

CONFIDENCE_THRESHOLD = 75.0
BET_SIZE_USDC        = 1.0
CLOB_HOST            = "https://clob.polymarket.com"
GAMMA_API            = "https://gamma-api.polymarket.com"
CHAIN_ID             = 137


class PolymarketTrader:

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

        self.client           = None  # initialised in async setup()
        self.active_bets      = []
        self.completed_bets   = []
        self.bets_file        = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "polymarket_bets.json"
        )
        self._last_bet_candle = 0
        self._wallet_balance  = 0.0
        self._cycle_count     = 0
        self._tor_available   = False
        self._load_bets()

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
        """Initialise the CLOB client, configure Tor proxy, fetch opening balance."""
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

                    try:
                        ob       = await asyncio.to_thread(self.client.get_order_book, target_token)
                        asks     = ob.asks if hasattr(ob, "asks") else []
                        best_ask = float(asks[0].price) if asks else 0.5
                    except Exception:
                        prices = market.get("outcomePrices", '["0.5","0.5"]')
                        if isinstance(prices, str):
                            prices = _json.loads(prices)
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

    async def place_bet(self, direction: str, confidence: float) -> dict | None:
        """
        Place a bet on the given direction if confidence ≥ threshold
        and we haven't already bet on the current 15-min candle.
        """
        if confidence < CONFIDENCE_THRESHOLD:
            log.debug(f"[PolyBet] Confidence {confidence:.1f}% < {CONFIDENCE_THRESHOLD}% — skip")
            return None

        current_candle = int(time.time()) // 900 * 900
        if current_candle == self._last_bet_candle:
            log.debug("[PolyBet] Already bet this candle — skipping")
            return None

        # Set immediately before any async work to prevent race conditions
        self._last_bet_candle = current_candle

        if self.client is None:
            log.warning("[PolyBet] CLOB client not initialised — skip")
            return None

        try:
            market = await self.find_btc_15min_market(direction)
            if not market:
                return None

            market_id = market["market_id"]
            token_id  = market["token_id"]
            question  = market["question"]
            end_time  = market["end_time"]
            best_ask  = market["best_ask"]

            if best_ask > 0.95 or best_ask < 0.05:
                log.warning(
                    f"[PolyBet] No value — odds {best_ask:.3f} indicate market already resolved, skipping"
                )
                return None

            shares = max(5.0, math.ceil((1.0 / best_ask) * 100) / 100)
            cost   = round(shares * best_ask, 4)

            order_args = OrderArgs(
                token_id=token_id,
                price=round(best_ask, 4),
                size=shares,
                side="BUY",
            )

            if self._use_tor:
                # Route order through torsocks subprocess so only this call
                # goes via Tor — the parent process network is not affected.
                _proj = os.path.dirname(os.path.abspath(__file__))
                _py   = sys.executable
                _script = (
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
                    f"token_id={token_id!r}, price={round(best_ask, 4)}, "
                    f"size={shares}, side='BUY'))\n"
                    f"resp = client.post_order(order, OrderType.GTC)\n"
                    f"print(json.dumps(resp))\n"
                )
                proc = await asyncio.to_thread(
                    subprocess.run,
                    ["torsocks", _py, "-c", _script],
                    capture_output=True, text=True, timeout=30,
                )
                if proc.returncode == 0:
                    response = json.loads(proc.stdout.strip())
                else:
                    raise Exception(f"torsocks order failed: {proc.stderr.strip()}")
            else:
                signed_order = await asyncio.to_thread(self.client.create_order, order_args)
                response     = await asyncio.to_thread(
                    self.client.post_order, signed_order, OrderType.GTC
                )

            # Detect geo-restriction in error responses before treating as success
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

            order_id = None
            if isinstance(response, dict):
                order_id = (
                    response.get("orderID")
                    or response.get("order_id")
                    or response.get("id")
                )

            bet = {
                "id":               str(order_id or f"local_{current_candle}"),
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
            }
            self.active_bets.append(bet)
            self._save_bets()

            log.info(
                f"[PolyBet] PLACED ${cost} on {direction} "
                f"| Market: {question} "
                f"| Odds: {best_ask:.3f} "
                f"| Shares: {shares} "
                f"| Potential profit: ${round(shares - cost, 4)}"
            )
            await self._refresh_balance()
            return bet

        except Exception as e:
            log.error(f"[PolyBet] place_bet failed: {e}", exc_info=True)
            return None

    # ── Resolution + claiming ────────────────────────────────────────

    async def check_and_claim(self):
        """Check resolved bets, record outcome, attempt redemption."""
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

                if now < end_dt:
                    still_active.append(bet)
                    continue

                async with httpx.AsyncClient(timeout=10) as hx:
                    resp = await hx.get(f"{GAMMA_API}/markets/{bet['market_id']}")

                if resp.status_code != 200:
                    still_active.append(bet)
                    continue

                market_data = resp.json()
                if not market_data.get("resolved", False):
                    still_active.append(bet)
                    continue

                resolved_outcome = str(
                    market_data.get("winner")
                    or market_data.get("resolvedOutcome")
                    or market_data.get("outcome", "")
                ).lower()

                direction = bet["direction"]
                won = (
                    direction == "UP"
                    and any(kw in resolved_outcome for kw in ("yes", "higher", "above", "up"))
                ) or (
                    direction == "DOWN"
                    and any(kw in resolved_outcome for kw in ("no", "lower", "below", "down"))
                )

                if won:
                    try:
                        if self.client and hasattr(self.client, "redeem_positions"):
                            await asyncio.to_thread(
                                self.client.redeem_positions, bet["token_id"]
                            )
                    except Exception as ce:
                        log.debug(f"[PolyBet] Redemption call (may auto-settle): {ce}")

                    actual_pnl = round(bet["shares"] - bet.get("bet_cost", bet["bet_usdc"]), 4)
                    bet.update({"status": "won", "result": "won", "pnl": actual_pnl})
                    log.info(f"[PolyBet] WON +${actual_pnl:.4f} | {bet['question'][:60]}")
                else:
                    loss = -round(bet.get("bet_cost", bet["bet_usdc"]), 4)
                    bet.update({"status": "lost", "result": "lost", "pnl": loss})
                    log.info(f"[PolyBet] LOST ${loss} | {bet['question'][:60]}")

                self.completed_bets.append(bet)
                self._save_bets()
                await self._refresh_balance()

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

    def _save_bets(self):
        try:
            with open(self.bets_file, "w") as f:
                json.dump(
                    {"active": self.active_bets, "completed": self.completed_bets},
                    f, indent=2,
                )
        except Exception as e:
            log.warning(f"[PolymarketTrader] Failed to save bets: {e}")

    # ── Status ───────────────────────────────────────────────────────

    def get_status(self) -> dict:
        total_won     = len([b for b in self.completed_bets if b.get("result") == "won"])
        total_lost    = len([b for b in self.completed_bets if b.get("result") == "lost"])
        total_settled = total_won + total_lost
        total_pnl     = sum(b["pnl"] for b in self.completed_bets if b.get("pnl") is not None)
        win_rate      = round(total_won / total_settled * 100, 1) if total_settled > 0 else 0.0
        return {
            "balance_usdc":   round(self._wallet_balance, 2),
            "active_bets":    self.active_bets,
            "completed_bets": self.completed_bets[-20:],
            "total_bets":     len(self.active_bets) + len(self.completed_bets),
            "total_won":      total_won,
            "total_lost":     total_lost,
            "total_pnl":      round(total_pnl, 4),
            "win_rate":       win_rate,
        }

    # ── Background run loop ──────────────────────────────────────────

    async def run_loop(self):
        """Runs every 60 s: checks resolutions, refreshes balance every 5 cycles."""
        while True:
            try:
                await self.check_and_claim()
                self._cycle_count += 1
                if self._cycle_count % 5 == 0:
                    await self._refresh_balance()
            except Exception as e:
                log.error(f"[PolymarketTrader] run_loop error: {e}")
            await asyncio.sleep(60)
