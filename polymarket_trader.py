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

CONFIDENCE_THRESHOLD = 75.0
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
    # 5. $1.01 per bet, 5 shares minimum

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
        self.bets_file      = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "polymarket_bets.json"
        )
        self._bet_candles   = set()  # tracks every candle that has been bet — never cleared
        self._wallet_balance = 0.0
        self._cycle_count    = 0
        self._tor_available  = False
        self._tor_proxies    = {}
        self._use_tor        = False
        self._load_bets()  # also rebuilds _bet_candles from history

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

    async def place_bet(self, direction: str, confidence: float) -> dict | None:
        try:
            # Guard: confidence threshold
            if confidence < CONFIDENCE_THRESHOLD:
                log.info(f"[PolyBet] Confidence {confidence:.1f}% below threshold — skip")
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

            # Get best ask from orderbook, fallback to outcomePrices
            try:
                ob       = await asyncio.to_thread(self.client.get_order_book, token_id)
                asks     = ob.asks if hasattr(ob, "asks") else []
                best_ask = float(asks[0].price) if asks else market["best_ask"]
            except Exception:
                best_ask = market["best_ask"]

            # Validate price — skip if market already nearly resolved
            if best_ask > 0.90 or best_ask < 0.10:
                log.warning(
                    f"[PolyBet] No value — odds {best_ask:.3f} market nearly resolved, skip"
                )
                return None

            # Calculate shares — FOK market order
            shares = max(5.0, math.ceil((1.0 / best_ask) * 100) / 100)
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
                _shares_retry = max(5.0, math.ceil((1.0 / _price_retry) * 100) / 100)
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
            log.error(f"[PolyBet] place_bet crashed: {e}")
            return None

    # ── Force resolve (ENFORCER) ─────────────────────────────────────

    async def _force_resolve_after(self, bet: dict, delay_seconds: int = 960):
        """
        Fires 16 minutes after bet placement.
        Forces resolution check and claim attempt.
        If market not yet resolved, retries every 60s for up to 10 minutes.
        Acts as hard safety net — no bet stays unresolved past 26 minutes.
        """
        await asyncio.sleep(delay_seconds)

        log.info(f"[PolyBet] ⏰ Force resolve triggered for: {bet['question'][:50]}")

        # Skip if already resolved by the check_and_claim() sweep
        if any(b["id"] == bet["id"] for b in self.completed_bets):
            log.info("[PolyBet] Already resolved by check_and_claim — skipping force resolve")
            return

        import requests as _req

        for attempt in range(10):
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

                    up_kw   = ["up", "yes", "higher", "above"]
                    down_kw = ["down", "no", "lower", "below"]

                    won = False
                    if bet["direction"] == "UP":
                        won = any(kw in winning_outcome for kw in up_kw)
                    elif bet["direction"] == "DOWN":
                        won = any(kw in winning_outcome for kw in down_kw)

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

                    if bet in self.active_bets:
                        self.active_bets.remove(bet)
                    self.completed_bets.append(bet)
                    self._save_bets()
                    await self._refresh_balance()
                    return

                else:
                    log.info(
                        f"[PolyBet] Market not yet resolved "
                        f"(attempt {attempt + 1}/10) — retry in 60s"
                    )
                    await asyncio.sleep(60)

            except Exception as e:
                log.error(f"[PolyBet] Force resolve attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(60)

        # 10 attempts exhausted — mark as unknown
        log.warning(
            f"[PolyBet] ⚠ Could not resolve after 10 attempts: "
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

                    self.completed_bets.append(bet)
                    self._save_bets()
                    await self._refresh_balance()

                elif seconds_past > 1800:
                    # 30 minutes past end_time, still unresolved — force-expire
                    log.warning(
                        f"[PolyBet] ⚠ Bet {bet['id'][:20]}… unresolved 30min+ past end_time "
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

                # Always sweep for resolved bets
                await self.check_and_claim()

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
