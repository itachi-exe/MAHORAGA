import os, sys, json, asyncio, logging, re, time, secrets

# Resolve runtime directory — works both normally and inside a PyInstaller binary
# Bundled files (html, .env) live in _MEIPASS; model files live next to the executable
_BUNDLE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
_APP_DIR    = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) \
              else os.path.dirname(os.path.abspath(__file__))
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional

try:
    import anthropic
except Exception:
    anthropic = None
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from core_trading_system import MAHORAGA, fetch_bybit_data, get_bybit_client, preprocess_data

# Load external .env (saved by wizard) first, fall back to bundled defaults
_EXT_ENV = os.path.join(_APP_DIR, '.env')
_BUN_ENV = os.path.join(_BUNDLE_DIR, '.env')
load_dotenv(dotenv_path=_BUN_ENV, override=False)   # bundled defaults
load_dotenv(dotenv_path=_EXT_ENV, override=True)    # client keys always win

API_KEY            = os.getenv('BYBIT_API_KEY', '')
API_SECRET         = os.getenv('BYBIT_API_SECRET', '')
ANTHROPIC_API_KEY  = os.getenv('ANTHROPIC_API_KEY', '')
DASHBOARD_PASSWORD = os.getenv('DASHBOARD_PASSWORD', os.getenv('CHAT_PASSWORD', ''))
SETTINGS_PASSWORD  = os.getenv('SETTINGS_PASSWORD', '')
BIND_HOST          = os.getenv('BIND_HOST', '127.0.0.1')
BIND_PORT          = int(os.getenv('BIND_PORT', '8501'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('MAHORAGA')

if not DASHBOARD_PASSWORD:
    log.warning('⚠  DASHBOARD_PASSWORD is not set — dashboard is UNPROTECTED')

# ── SESSION STORE ─────────────────────────────────────────────────
_sessions: dict[str, float] = {}    # token → expiry timestamp
SESSION_TTL    = 8 * 3600           # 8 hours
SESSION_COOKIE = 'mhg_session'
MAX_SESSIONS   = 50                 # evict oldest if exceeded

def _make_token() -> str:
    return secrets.token_hex(32)

def _valid_session(token: str | None) -> bool:
    if not token:
        return False
    exp = _sessions.get(token)
    if exp is None:
        return False
    if time.time() > exp:
        _sessions.pop(token, None)
        return False
    _sessions[token] = time.time() + SESSION_TTL   # rolling TTL
    return True

def _create_session() -> str:
    # Evict oldest sessions if at capacity
    if len(_sessions) >= MAX_SESSIONS:
        oldest = min(_sessions, key=_sessions.get)
        _sessions.pop(oldest, None)
    token = _make_token()
    _sessions[token] = time.time() + SESSION_TTL
    return token

def _revoke_session(token: str | None):
    if token:
        _sessions.pop(token, None)

# ── AUTH DEPENDENCY ────────────────────────────────────────────────
async def require_auth(request: Request):
    token = (
        request.cookies.get(SESSION_COOKIE) or
        request.headers.get('X-Session-Token', '')
    )
    if DASHBOARD_PASSWORD and not _valid_session(token):
        raise HTTPException(status_code=401, detail='Unauthorised — please login')

# ── RATE LIMITING ─────────────────────────────────────────────────
CHAT_RATE_LIMIT  = 15   # per minute per IP
ORDER_RATE_LIMIT = 5    # per minute per IP
API_RATE_LIMIT   = 60   # per minute per IP for generic endpoints
MAX_MSG_LEN      = 2000 # max characters per chat message
_rate: dict = defaultdict(lambda: defaultdict(list))

def _check_rate(ip: str, kind: str, limit: int) -> bool:
    """Return True if rate limit is exceeded."""
    now = time.time()
    _rate[kind][ip] = [t for t in _rate[kind][ip] if now - t < 60]
    if len(_rate[kind][ip]) >= limit:
        return True
    _rate[kind][ip].append(now)
    return False

_STRIP_HTML = re.compile(r'<[^>]+>')
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

# ── CHAT AI SYSTEM ────────────────────────────────────────────────
_SECURITY_SYSTEM = """\
You are MAHORAGA, a read-only AI trading assistant for a live Bybit futures account.

ALLOWED:
- Answer questions about account balance, equity, P&L, positions, signals, market data, bot status
- Explain what the auto-trader is doing
- Close the user's open position when they clearly and explicitly ask

STRICTLY FORBIDDEN — refuse immediately if asked:
- Changing or suggesting changes to: stop loss %, take profit %, trailing stop %, confidence threshold, max daily loss, trade size, leverage, or any risk parameter
- Starting or stopping the auto-trader
- Placing, entering, or recommending new trade entries
- Switching trading pairs or intervals
- Providing price predictions or financial advice
- Giving instructions that would let someone manually perform any of the above

You have exactly ONE action available: close_position.
Use it ONLY when the user explicitly says to close their open position.
For everything else, answer in plain text only. Be concise.
"""

_CLOSE_TOOL = {
    "name": "close_position",
    "description": "Closes the user's currently open BTC position on Bybit with a reduce-only market order. Only call this when the user explicitly asks to close their open position.",
    "input_schema": {"type": "object", "properties": {}, "required": []}
}


def _require_anthropic():
    if anthropic is None:
        raise HTTPException(
            status_code=503,
            detail="Anthropic integration is disabled (package not installed)."
        )

# ── APP ───────────────────────────────────────────────────────────
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)  # disable public docs
bot = MAHORAGA()

# ── CORS ──────────────────────────────────────────────────────────
_allowed_origins = [
    f"http://localhost:{BIND_PORT}",
    f"http://127.0.0.1:{BIND_PORT}",
    f"http://{BIND_HOST}:{BIND_PORT}",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-Session-Token", "Content-Type"],
)

# ── SECURITY HEADERS MIDDLEWARE ────────────────────────────────────
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"]  = "nosniff"
    response.headers["X-Frame-Options"]          = "DENY"
    response.headers["X-XSS-Protection"]         = "1; mode=block"
    response.headers["Referrer-Policy"]           = "no-referrer"
    response.headers["Cache-Control"]             = "no-store"
    response.headers["Permissions-Policy"]        = "geolocation=(), microphone=(), camera=()"
    return response

# ── AUTO TRADER ────────────────────────────────────────────────────
class AutoTrader:
    SYMBOL   = 'BTCUSDT'
    INTERVAL = '60'
    MIN_QTY  = 0.001

    # ── GHOST-IN-THE-MARKET RISK CONSTANTS (hardcoded, non-negotiable) ──
    RISK_PER_TRADE       = 0.01    # 1%  — fraction of balance risked per chunk
    CONF_THRESHOLD       = 0.65    # 65% — minimum model confidence to trade
    STOP_LOSS_PCT        = 1.0     # 1%  — stop-loss distance from entry
    TAKE_PROFIT_PCT      = 3.0     # 3%  — take-profit (3:1 reward:risk)
    TRAILING_STOP_PCT    = 1.0     # 1%  — trailing stop distance
    MAX_DAILY_LOSS_PCT   = 3.0     # 3%  — halt when daily loss reaches this
    MAX_DAILY_PROFIT_PCT = 10.0    # 10% — halt when daily profit reaches this
    MAX_TRADES_PER_DAY   = 10      # hard cap on total trades per day
    MAX_CONSEC_LOSSES    = 3       # halt after this many consecutive losses
    MAX_CHUNKS           = 3       # max position entries in the same direction

    def __init__(self):
        self.running              = False
        self.task                 = None
        self.check_secs           = 120
        self.first_delay_secs     = 3
        self.retry_delay_secs     = 12
        self.last_signal          = None
        self.next_check           = None
        self.trade_log            = []
        # ── Daily state ───────────────────────────────────────────────
        self.trading_day          = None
        self.trades_today         = 0
        self.consec_losses        = 0
        self.daily_start_balance  = None
        # ── Direction state (Ghost algorithm) ─────────────────────────
        self.current_day_direction   = None  # direction locked in for today
        self.previous_day_direction  = None  # direction from yesterday (conflict resolution)
        self.current_position_entries = 0
        # ── 5-hour cooloff between trades ─────────────────────────────
        self.last_trade_time         = None  # datetime of last successfully opened trade

    def _client(self):
        return get_bybit_client(API_KEY, API_SECRET)

    def _get_position(self, client):
        resp = client.get_positions(category='linear', symbol=self.SYMBOL)
        for p in resp['result']['list']:
            if float(p.get('size', 0)) > 0:
                return p
        return None

    def _get_qty(self, client):
        try:
            w     = client.get_wallet_balance(accountType='UNIFIED', coin='USDT')
            bal   = float(w['result']['list'][0]['coin'][0]['walletBalance'])
            t     = client.get_tickers(category='linear', symbol=self.SYMBOL)
            price = float(t['result']['list'][0]['lastPrice'])
            # Ghost algorithm: risk exactly RISK_PER_TRADE% of balance per chunk.
            # position_value = (balance × risk%) / sl%
            # This stays correct regardless of what SL% is set to.
            risk_usdt    = bal * self.RISK_PER_TRADE
            position_val = risk_usdt / (self.STOP_LOSS_PCT / 100)
            qty = round(position_val / price, 3)
            return max(self.MIN_QTY, qty)
        except Exception:
            return self.MIN_QTY

    def _get_wallet(self, client):
        w = client.get_wallet_balance(accountType='UNIFIED', coin='USDT')
        coin_list = w.get('result', {}).get('list', [{}])
        coins = coin_list[0].get('coin', [{}]) if coin_list else [{}]
        return float((coins[0] if coins else {}).get('walletBalance', 0))

    def _sl_tp_prices(self, client, side):
        t     = client.get_tickers(category='linear', symbol=self.SYMBOL)
        price = float(t['result']['list'][0]['markPrice'])
        sl_dist = price * (self.STOP_LOSS_PCT / 100)
        tp_dist = price * (self.TAKE_PROFIT_PCT / 100)
        if side == 'Buy':
            return round(price - sl_dist, 2), round(price + tp_dist, 2)
        else:
            return round(price + sl_dist, 2), round(price - tp_dist, 2)

    def _trailing_distance(self, client):
        t     = client.get_tickers(category='linear', symbol=self.SYMBOL)
        price = float(t['result']['list'][0]['markPrice'])
        return str(round(price * (self.TRAILING_STOP_PCT / 100), 2))

    def _place(self, client, side, qty, reduce_only=False, sl=None, tp=None, trailing=None):
        kwargs = dict(
            category='linear', symbol=self.SYMBOL,
            side=side, orderType='Market',
            qty=str(qty), timeInForce='IOC',
        )
        if reduce_only:
            kwargs['reduceOnly'] = True
        if sl:
            kwargs['stopLoss']      = str(sl)
            kwargs['slTriggerBy']   = 'MarkPrice'
        if tp:
            kwargs['takeProfit']    = str(tp)
            kwargs['tpTriggerBy']   = 'MarkPrice'
        if trailing:
            kwargs['trailingStop']  = str(trailing)
        return client.place_order(**kwargs)

    def _check_daily_limits(self, client):
        try:
            now       = datetime.now(timezone.utc)
            today_str = now.strftime('%Y-%m-%d')
            bal       = self._get_wallet(client)

            # ── Day rollover ──────────────────────────────────────────
            if self.trading_day != today_str:
                # Ghost algorithm: carry direction forward for conflict detection
                # on the new day's first signal
                self.previous_day_direction  = self.current_day_direction
                self.current_day_direction   = None
                self.trading_day             = today_str
                self.trades_today            = 0
                self.consec_losses           = 0
                self.daily_start_balance     = bal
                self.current_position_entries = 0
                self.last_trade_time         = None  # reset cooloff on new day
                return False, ""

            if self.daily_start_balance is None:
                self.daily_start_balance = bal

            # ── Daily PNL limits ──────────────────────────────────────
            if self.daily_start_balance > 0:
                pnl_pct = (bal - self.daily_start_balance) / self.daily_start_balance * 100
                if pnl_pct <= -self.MAX_DAILY_LOSS_PCT:
                    return True, f"Max daily loss -{self.MAX_DAILY_LOSS_PCT}% reached: {round(pnl_pct, 2)}%"
                if pnl_pct >= self.MAX_DAILY_PROFIT_PCT:
                    return True, f"Daily profit target +{self.MAX_DAILY_PROFIT_PCT}% reached: {round(pnl_pct, 2)}%"

            # ── Re-verify counts from exchange API ────────────────────
            today_ms = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
            try:
                er = client.get_closed_pnl(category='linear', symbol=self.SYMBOL,
                                           startTime=str(today_ms), limit=50)
                closed_trades = er.get('result', {}).get('list', [])
                self.trades_today = len(closed_trades)
                # Consecutive losses — list is newest-first
                l_count = 0
                for trade in closed_trades:
                    if float(trade.get('closedPnl', 0)) < 0:
                        l_count += 1
                    else:
                        break
                self.consec_losses = l_count
            except Exception as inner_e:
                log.warning(f"Failed to fetch closed PNL for limits: {inner_e}")

            if self.trades_today >= self.MAX_TRADES_PER_DAY:
                return True, f"Max {self.MAX_TRADES_PER_DAY} trades/day reached."
            if self.consec_losses >= self.MAX_CONSEC_LOSSES:
                return True, f"{self.MAX_CONSEC_LOSSES} consecutive losses reached."

            return False, ""
        except Exception as e:
            log.error(f"Error checking daily limits: {e}")
            return False, ""

    def _log(self, signal, confidence, action, note=''):
        entry = {
            'time':       datetime.now().strftime('%H:%M:%S'),
            'signal':     signal,
            'confidence': round(float(confidence) * 100, 1),
            'action':     action,
            'note':       note,
        }
        self.trade_log.insert(0, entry)
        self.trade_log = self.trade_log[:100]
        log.info(f"[AutoTrader] {action} | {signal} {entry['confidence']}% | {note}")

    async def run(self):
        first_cycle = True
        while self.running:
            # First decision cycle runs quickly after start, then normal cadence.
            cycle_delay = self.first_delay_secs if first_cycle else self.check_secs
            self.next_check = datetime.now().timestamp() + cycle_delay
            try:
                client = self._client()

                position = self._get_position(client)
                if position and not position.get('stopLoss') and not position.get('takeProfit'):
                    try:
                        side  = position['side']
                        sl, tp = self._sl_tp_prices(client, side)
                        trail  = self._trailing_distance(client)
                        client.set_trading_stop(
                            category='linear', symbol=self.SYMBOL,
                            stopLoss=str(sl), takeProfit=str(tp),
                            trailingStop=trail,
                            slTriggerBy='MarkPrice', tpTriggerBy='MarkPrice',
                            positionIdx=0
                        )
                        self._log('GUARD', 0, f'PROTECTED EXISTING {side}',
                                  f'SL=${sl} TP=${tp} Trail={self.TRAILING_STOP_PCT}%')
                    except Exception as eg:
                        self._log('GUARD', 0, 'FAILED TO PROTECT POSITION', str(eg)[:80])

                limit_hit, limit_msg = self._check_daily_limits(client)
                if limit_hit:
                    self._log('RISK', 0, 'STOPPED FOR THE DAY', limit_msg)
                    self.stop()
                    break

                df       = await asyncio.to_thread(
                                fetch_bybit_data, symbol=self.SYMBOL, interval=self.INTERVAL,
                                limit=300, api_key=API_KEY, api_secret=API_SECRET)
                features = await asyncio.to_thread(bot.preprocess_single_bar, df.copy())
                signal, confidence = await asyncio.to_thread(bot.predict, features, self.CONF_THRESHOLD)
                self.last_signal = {
                    'signal':     signal,
                    'confidence': round(float(confidence), 4),
                    'time':       datetime.now().strftime('%H:%M:%S'),
                }

                position = self._get_position(client)
                pos_side = position['side'] if position else None

                # ── GHOST: Confidence filter ──────────────────────────
                if confidence < self.CONF_THRESHOLD or signal == 'HOLD':
                    reason = f"signal={signal}, conf={round(confidence*100,1)}%, threshold={round(self.CONF_THRESHOLD*100,1)}%"
                    if signal == 'HOLD':
                        reason += " (model says HOLD)"
                    else:
                        reason += " (below threshold)"
                    self._log(signal, confidence, 'IGNORED', reason)

                elif signal in ('BUY', 'SELL'):
                    target_side = 'Buy' if signal == 'BUY' else 'Sell'

                    # ── GHOST: Previous-day conflict resolution ───────
                    # If carry-over position exists from yesterday in the opposite
                    # direction, close it before establishing today's direction.
                    if pos_side and self.previous_day_direction and signal != self.previous_day_direction:
                        close_side = 'Sell' if pos_side == 'Buy' else 'Buy'
                        cr = self._place(client, close_side, float(position['size']), reduce_only=True)
                        if cr.get('retCode') == 0:
                            self._log(signal, confidence, f'CLOSED PREV-DAY {pos_side.upper()}',
                                      f'yesterday={self.previous_day_direction}, today signal={signal}')
                            self.previous_day_direction   = None
                            self.current_position_entries = 0
                            pos_side  = None
                            position  = None
                            await asyncio.sleep(1)
                        else:
                            self._log(signal, confidence, 'FAILED',
                                      f"prev-day close: {cr.get('retMsg')}")
                            await asyncio.sleep(self.check_secs); continue

                    # ── GHOST: Direction lock — no intra-day flipping ─
                    # First trade of the day establishes the direction.
                    # All opposite signals during the same day are ignored.
                    if self.current_day_direction is None:
                        self.current_day_direction = signal
                    elif signal != self.current_day_direction:
                        self._log(signal, confidence, 'IGNORED',
                                  f'opposite direction — locked {self.current_day_direction} for today')
                        await asyncio.sleep(self.check_secs); continue

                    # ── GHOST: Max chunks in same direction ───────────
                    if not pos_side:
                        self.current_position_entries = 0

                    if pos_side == target_side:
                        if self.current_position_entries >= self.MAX_CHUNKS:
                            self._log(signal, confidence, 'IGNORED',
                                      f'max {self.MAX_CHUNKS} chunks in {target_side} reached')
                            await asyncio.sleep(self.check_secs); continue
                        self._log(signal, confidence, 'ADDING',
                                  f'chunk {self.current_position_entries + 1}/{self.MAX_CHUNKS}')

                    # ── GHOST: 5-hour cooloff between trades ─────────────
                    current_time = datetime.now()
                    if self.last_trade_time is not None:
                        elapsed = (current_time - self.last_trade_time).total_seconds()
                        if elapsed < 5 * 3600:
                            remaining = int(5 * 3600 - elapsed)
                            h, m = remaining // 3600, (remaining % 3600) // 60
                            self._log(signal, confidence, 'IGNORED',
                                      f'STOP: Cooloff period active — {h}h {m}m remaining')
                            await asyncio.sleep(self.check_secs)
                            continue

                    # ── GHOST: Claude API trade confirmation ──────────────
                    if anthropic is not None and ANTHROPIC_API_KEY:
                        try:
                            # Gather market context for Claude
                            rsi_val = None
                            try:
                                rsi_val = round(float(features['RSI'].iloc[-1]), 2) \
                                          if 'RSI' in features.columns else None
                            except Exception:
                                pass
                            last_price = round(float(df['close'].iloc[-1]), 2)
                            price_24h_ago = round(float(df['close'].iloc[-24]) if len(df) >= 24 else float(df['close'].iloc[0]), 2)
                            trend = 'uptrend' if last_price > price_24h_ago else 'downtrend'
                            confirm_payload = json.dumps({
                                'direction':   signal,
                                'confidence':  round(float(confidence), 4),
                                'rsi':         rsi_val,
                                'btc_price':   last_price,
                                'market_trend': trend,
                            })
                            ac = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                            confirm_resp = await asyncio.to_thread(
                                ac.messages.create,
                                model='claude-3-haiku-20240307',
                                max_tokens=200,
                                system=(
                                    'You are a strict ICT trading risk manager for a BTC bot. '
                                    'Analyze the trade signal and respond with JSON only: '
                                    '{"approved": true/false, "reason": string, "adjusted_confidence": number}. '
                                    'Only approve trades with strong ICT confluence.'
                                ),
                                messages=[{'role': 'user', 'content': confirm_payload}]
                            )
                            ai_text = next((b.text for b in confirm_resp.content
                                           if hasattr(b, 'text')), '{}')
                            m_json = _JSON_BLOCK.search(ai_text)
                            claude_decision = json.loads(m_json.group(0)) if m_json else {}
                            if not claude_decision.get('approved', False):
                                reason = str(claude_decision.get('reason', 'no reason'))[:120]
                                self._log(signal, confidence, 'IGNORED',
                                          f'IGNORED: Claude rejected trade — {reason}')
                                await asyncio.sleep(self.check_secs)
                                continue
                            log.info(f'[AutoTrader] Claude approved {signal} trade — '
                                     f'{claude_decision.get("reason","")[:80]}')
                        except Exception as claude_err:
                            log.warning(f'[AutoTrader] Claude API error: {claude_err}. '
                                        'Proceeding without Claude confirmation.')

                    # ── Execute trade ─────────────────────────────────
                    sl, tp = self._sl_tp_prices(client, target_side)
                    trail  = self._trailing_distance(client)
                    qty    = self._get_qty(client)
                    r      = self._place(client, target_side, qty, sl=sl, tp=tp, trailing=trail)
                    if r.get('retCode') == 0:
                        self.current_position_entries += 1
                        self.last_trade_time = datetime.now()   # start 5hr cooloff
                        self._log(signal, confidence,
                                  f'OPENED {target_side.upper()} {qty} BTC (chunk {self.current_position_entries}/{self.MAX_CHUNKS})',
                                  f'SL=${sl} TP=${tp} Trail={self.TRAILING_STOP_PCT}% | {r["result"].get("orderId","")[:8]}')
                    else:
                        self._log(signal, confidence, f'FAILED {target_side.upper()}',
                                  f"retCode={r.get('retCode')} msg={r.get('retMsg', '')}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                msg = str(e)
                self._log('ERROR', 0, 'EXCEPTION', msg[:120])
                transient = (
                    "HTTPSConnectionPool" in msg
                    or "Max retries exceeded" in msg
                    or "Read timed out" in msg
                    or "ConnectTimeout" in msg
                    or "Connection aborted" in msg
                )
                if transient:
                    self._log('ERROR', 0, 'RETRYING', f'network issue, retry in {self.retry_delay_secs}s')
                    await asyncio.sleep(self.retry_delay_secs)
                    first_cycle = False
                    continue

            first_cycle = False
            await asyncio.sleep(cycle_delay)

    def start(self):
        if not self.running:
            self.running         = True
            self._daily_loss_ref = None
            self._log('SYSTEM', 0, 'STARTED',
                      f'first check in {self.first_delay_secs}s, then every {self.check_secs}s')
            self.task            = asyncio.create_task(self.run())
            log.info('[AutoTrader] Started')

    def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            self.task = None
        log.info('[AutoTrader] Stopped')

    def status(self):
        secs_left = max(0, int((self.next_check or 0) - datetime.now().timestamp())) if self.running else 0
        now = datetime.now()
        cooloff_remaining = 0
        if self.last_trade_time is not None:
            elapsed = (now - self.last_trade_time).total_seconds()
            cooloff_remaining = max(0, int(5 * 3600 - elapsed))
        return {
            'running':               self.running,
            'symbol':                self.SYMBOL,
            'interval':              self.INTERVAL,
            'check_secs':            self.check_secs,
            'next_check':            secs_left,
            'last_signal':           self.last_signal,
            'current_day_direction': self.current_day_direction,
            'trades_today':          self.trades_today,
            'consec_losses':         self.consec_losses,
            'last_trade_time':       self.last_trade_time.strftime('%H:%M:%S') if self.last_trade_time else None,
            'cooloff_remaining_secs': cooloff_remaining,
            'log':                   self.trade_log[:20],
            # Hardcoded Ghost-algorithm constants (read-only)
            'risk_constants': {
                'risk_per_trade_pct':   self.RISK_PER_TRADE * 100,
                'conf_threshold_pct':   self.CONF_THRESHOLD * 100,
                'stop_loss_pct':        self.STOP_LOSS_PCT,
                'take_profit_pct':      self.TAKE_PROFIT_PCT,
                'trailing_stop_pct':    self.TRAILING_STOP_PCT,
                'max_daily_loss_pct':   self.MAX_DAILY_LOSS_PCT,
                'max_daily_profit_pct': self.MAX_DAILY_PROFIT_PCT,
                'max_trades_per_day':   self.MAX_TRADES_PER_DAY,
                'max_consec_losses':    self.MAX_CONSEC_LOSSES,
                'max_chunks':           self.MAX_CHUNKS,
            }
        }

autotrader = AutoTrader()


def _parse_ai_trade_plan(raw_text: str) -> dict:
    """Parse Anthropic JSON response for hybrid trade proposals."""
    txt = (raw_text or "").strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt.replace("json", "", 1).strip()
    m = _JSON_BLOCK.search(txt)
    if m:
        txt = m.group(0)
    try:
        data = json.loads(txt)
    except Exception:
        return {"action": "HOLD", "confidence": 0.0, "reason": "invalid ai response format"}

    action = str(data.get("action", "HOLD")).upper()
    if action not in ("BUY", "SELL", "HOLD"):
        action = "HOLD"
    try:
        conf = float(data.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    reason = str(data.get("reason", "")).strip()[:240]
    return {"action": action, "confidence": conf, "reason": reason}

# ── AUTH ENDPOINTS ────────────────────────────────────────────────
class LoginRequest(BaseModel):
    password: str

@app.post("/api/auth/login")
async def auth_login(req: LoginRequest, request: Request, response: Response):
    ip = request.client.host
    if _check_rate(ip, 'login', 5):  # 5 attempts per minute max
        log.warning(f"[Auth] Login rate limit hit from {ip}")
        return JSONResponse(status_code=429, content={"error": "Too many login attempts."})

    # Constant-time comparison to prevent timing attacks
    if not DASHBOARD_PASSWORD or not secrets.compare_digest(req.password, DASHBOARD_PASSWORD):
        log.warning(f"[Auth] Failed login attempt from {ip}")
        await asyncio.sleep(1)  # Slow down brute force
        return JSONResponse(status_code=401, content={"error": "Invalid password."})

    token = _create_session()
    log.info(f"[Auth] Login from {ip}")
    resp = JSONResponse(content={"ok": True})
    resp.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="strict",
        secure=False,          # set True if using HTTPS
        max_age=SESSION_TTL,
        path="/",
    )
    return resp

@app.get("/api/auth/check")
async def auth_check(request: Request):
    token = request.cookies.get(SESSION_COOKIE) or request.headers.get('X-Session-Token', '')
    if DASHBOARD_PASSWORD and not _valid_session(token):
        return JSONResponse(status_code=401, content={"authenticated": False})
    return {"authenticated": True}

@app.post("/api/auth/logout")
async def auth_logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    _revoke_session(token)
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp

# ── Settings password verification ────────────────────────────────
class SettingsAuthRequest(BaseModel):
    password: str

@app.post("/api/settings/auth")
async def settings_auth(req: SettingsAuthRequest, request: Request,
                        _auth=Depends(require_auth)):
    ip = request.client.host
    if _check_rate(ip, 'settings_auth', 5):
        return JSONResponse(status_code=429, content={"error": "Too many attempts."})
    if not SETTINGS_PASSWORD or not secrets.compare_digest(req.password, SETTINGS_PASSWORD):
        await asyncio.sleep(1)
        return JSONResponse(status_code=401, content={"error": "Wrong settings password."})
    return JSONResponse(content={"ok": True})

# ── Dashboard HTML ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(_BUNDLE_DIR, "MAHORAGA_dashboard.html")
    with open(html_path, "r") as f:
        return f.read()

# ── REST: market snapshot ──────────────────────────────────────────
@app.get("/api/market")
def market(symbol: str = "BTCUSDT", interval: str = "60", limit: int = 200,
           _auth=Depends(require_auth)):
    # Validate params
    if interval not in ('1','3','5','15','30','60','120','240','360','720','D','W','M'):
        raise HTTPException(400, "Invalid interval")
    limit = max(1, min(limit, 1000))
    try:
        df = fetch_bybit_data(symbol=symbol, interval=interval, limit=limit,
                              api_key=API_KEY, api_secret=API_SECRET)
        candles = []
        for ts, row in df.iterrows():
            candles.append({
                "time": int(ts.timestamp()),
                "open": row["open"], "high": row["high"],
                "low": row["low"],   "close": row["close"],
                "volume": row["volume"]
            })
        current = df["close"].iloc[-1]
        prev    = df["close"].iloc[-2]
        change  = round((current - prev) / prev * 100, 2)
        return {"candles": candles, "price": round(float(current), 2),
                "change": change,
                "high24": round(float(df["high"].tail(24).max()), 2),
                "low24":  round(float(df["low"].tail(24).min()), 2)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── REST: AI signal ────────────────────────────────────────────────
@app.get("/api/signal")
def signal(symbol: str = "BTCUSDT", interval: str = "60", confidence: float = 0.6,
           _auth=Depends(require_auth)):
    try:
        df = fetch_bybit_data(symbol=symbol, interval=interval, limit=300,
                              api_key=API_KEY, api_secret=API_SECRET)
        if not bot.model:
            return {"signal": "NO MODEL", "confidence": 0, "note": "Train model first"}
        features = bot.preprocess_single_bar(df.copy())
        sig, conf = bot.predict(features, confidence)
        return {"signal": sig, "confidence": round(float(conf), 4)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── REST: balance ──────────────────────────────────────────────────
@app.get("/api/balance")
def balance(_auth=Depends(require_auth)):
    try:
        bal = bot.get_balance(API_KEY, API_SECRET)
        return {"usdt": round(bal, 2)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── REST: full dashboard data ──────────────────────────────────────
@app.get("/api/dashboard")
def dashboard_data(symbol: str = "BTCUSDT", _auth=Depends(require_auth)):
    try:
        client = get_bybit_client(API_KEY, API_SECRET)

        w         = client.get_wallet_balance(accountType='UNIFIED', coin='USDT')
        coin_list = w.get('result', {}).get('list', [{}])
        coins     = coin_list[0].get('coin', [{}]) if coin_list else [{}]
        coin      = coins[0] if coins else {}

        pr = client.get_positions(category='linear', symbol=symbol)
        positions = []
        for p in pr['result']['list']:
            if float(p.get('size', 0)) > 0:
                positions.append({
                    'symbol':        p['symbol'],
                    'side':          p['side'],
                    'size':          float(p['size']),
                    'entryPrice':    float(p['avgPrice']),
                    'markPrice':     float(p['markPrice']),
                    'unrealisedPnl': float(p['unrealisedPnl']),
                    'leverage':      p['leverage'],
                    'liqPrice':      float(p['liqPrice']) if p.get('liqPrice') else 0,
                })

        today_ms  = int(datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        er        = client.get_executions(category='linear', limit=200)
        execs     = er['result']['list']
        today_ex  = [e for e in execs if int(e['execTime']) >= today_ms]
        today_pnl = sum(float(e.get('closedPnl', 0)) - float(e.get('execFee', 0)) for e in today_ex)

        trade_history = []
        for e in execs[:20]:
            trade_history.append({
                'symbol': e.get('symbol', ''),
                'side': e.get('side', ''),
                'execPrice': float(e.get('execPrice', 0)),
                'execQty': float(e.get('execQty', 0)),
                'execTime': int(e.get('execTime', 0)),
                'closedPnl': float(e.get('closedPnl', 0)) - float(e.get('execFee', 0))
            })

        return {
            'walletBalance':  round(float(coin.get('walletBalance',  0)), 4),
            'equity':         round(float(coin.get('equity',         0)), 4),
            'unrealisedPnl':  round(float(coin.get('unrealisedPnl',  0)), 4),
            'cumRealisedPnl': round(float(coin.get('cumRealisedPnl', 0)), 4),
            'todayPnl':       round(today_pnl, 4),
            'todayTrades':    len(today_ex),
            'totalTrades':    len(execs),
            'positions':      positions,
            'tradeHistory':   trade_history,
            'autotrader':     autotrader.status(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── REST: place order ──────────────────────────────────────────────
class OrderRequest(BaseModel):
    symbol: str = "BTCUSDT"
    side:   str
    qty:    float

    @field_validator('side')
    @classmethod
    def validate_side(cls, v):
        if v not in ('Buy', 'Sell'):
            raise ValueError("side must be 'Buy' or 'Sell'")
        return v

    @field_validator('qty')
    @classmethod
    def validate_qty(cls, v):
        if v <= 0 or v > 100:
            raise ValueError("qty must be between 0 and 100")
        return round(v, 3)

@app.post("/api/order")
def order(req: OrderRequest, request: Request, _auth=Depends(require_auth)):
    ip = request.client.host
    if _check_rate(ip, 'order', ORDER_RATE_LIMIT):
        return JSONResponse(status_code=429, content={"error": "Too many orders. Wait a moment."})
    try:
        result = bot.place_order(req.symbol, req.side, req.qty, API_KEY, API_SECRET)
        if result.get("retCode") != 0:
            msg = result.get("retMsg", "Unknown error")
            return JSONResponse(status_code=400, content={"error": msg, "retCode": result.get("retCode")})
        log.info(f"[Order] {req.side} {req.qty} {req.symbol} from {ip}")
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── REST: autotrader controls ──────────────────────────────────────
@app.post("/api/autotrader/start")
async def at_start(_auth=Depends(require_auth)):
    if not bot.model:
        return JSONResponse(status_code=400, content={"error": "No model loaded. Train first."})
    autotrader.start()
    return {"status": "started"}

@app.post("/api/autotrader/stop")
async def at_stop(_auth=Depends(require_auth)):
    autotrader.stop()
    return {"status": "stopped"}

@app.get("/api/autotrader/status")
async def at_status(_auth=Depends(require_auth)):
    return autotrader.status()

@app.post("/api/autotrader/settings")
async def at_settings(_auth=Depends(require_auth)):
    # All risk parameters are hardcoded Ghost-algorithm constants.
    # They cannot be changed at runtime — returns the current constants.
    return JSONResponse(status_code=423, content={
        "error": "Risk parameters are hardcoded (Ghost-in-the-Market algorithm). They cannot be changed at runtime.",
        "risk_constants": autotrader.status()['risk_constants'],
    })


@app.post("/api/trade/hybrid-entry")
async def hybrid_trade_entry(symbol: str = "BTCUSDT", interval: str = "60",
                             anthropic_min_conf: float = 0.55,
                             _auth=Depends(require_auth)):
    """
    Anthropic proposes a direction; model must confirm before execution.
    """
    if interval not in ('1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M'):
        raise HTTPException(400, "Invalid interval")
    if anthropic_min_conf < 0.0 or anthropic_min_conf > 1.0:
        raise HTTPException(400, "anthropic_min_conf must be between 0 and 1")
    if not API_KEY or not API_SECRET:
        raise HTTPException(400, "Bybit API keys are missing in .env")
    _require_anthropic()

    try:
        # 1) Market snapshot and model confirmation
        df = await asyncio.to_thread(
            fetch_bybit_data,
            symbol=symbol,
            interval=interval,
            limit=300,
            api_key=API_KEY,
            api_secret=API_SECRET
        )
        features = await asyncio.to_thread(bot.preprocess_single_bar, df.copy())
        model_signal, model_conf = await asyncio.to_thread(
            bot.predict, features, autotrader.CONF_THRESHOLD
        )

        if model_signal == "HOLD" or model_conf < autotrader.CONF_THRESHOLD:
            note = f"model blocked entry: signal={model_signal}, conf={round(model_conf*100,1)}%"
            autotrader._log(model_signal, model_conf, "HYBRID SKIP", note)
            return {
                "status": "skip",
                "reason": "Model did not confirm an entry",
                "model_signal": model_signal,
                "model_confidence": round(float(model_conf), 4),
            }

        # 2) Ask Anthropic for directional proposal (JSON only)
        recent = df.tail(25).reset_index()
        rows = []
        for _, r in recent.iterrows():
            rows.append({
                "t": str(r["timestamp"]),
                "o": round(float(r["open"]), 2),
                "h": round(float(r["high"]), 2),
                "l": round(float(r["low"]), 2),
                "c": round(float(r["close"]), 2),
                "v": round(float(r["volume"]), 3),
            })
        prompt = (
            "Analyze BTCUSDT market candles and return ONLY JSON with keys: "
            "action (BUY|SELL|HOLD), confidence (0..1), reason.\n"
            f"Symbol={symbol}, Interval={interval}, LastPrice={round(float(df['close'].iloc[-1]), 2)}\n"
            f"RecentCandles={json.dumps(rows)}"
        )
        ac = anthropic.Anthropic()
        ai_resp = await asyncio.to_thread(
            ac.messages.create,
            model="claude-sonnet-4-6",
            max_tokens=180,
            system="You are a trading signal classifier. Output strict JSON only.",
            messages=[{"role": "user", "content": prompt}]
        )
        ai_text = next((b.text for b in ai_resp.content if hasattr(b, "text")), "")
        ai_plan = _parse_ai_trade_plan(ai_text)

        # 3) Both sides must agree to execute
        if ai_plan["action"] not in ("BUY", "SELL"):
            autotrader._log(model_signal, model_conf, "HYBRID SKIP",
                            f"anthropic={ai_plan['action']} reason={ai_plan['reason']}")
            return {
                "status": "skip",
                "reason": "Anthropic did not propose BUY/SELL",
                "anthropic": ai_plan,
                "model_signal": model_signal,
                "model_confidence": round(float(model_conf), 4),
            }
        if ai_plan["confidence"] < anthropic_min_conf:
            autotrader._log(model_signal, model_conf, "HYBRID SKIP",
                            f"anthropic confidence too low: {round(ai_plan['confidence']*100,1)}%")
            return {
                "status": "skip",
                "reason": "Anthropic confidence too low",
                "anthropic": ai_plan,
                "model_signal": model_signal,
                "model_confidence": round(float(model_conf), 4),
            }
        if ai_plan["action"] != model_signal:
            autotrader._log(model_signal, model_conf, "HYBRID SKIP",
                            f"disagreement anthropic={ai_plan['action']} model={model_signal}")
            return {
                "status": "skip",
                "reason": "Anthropic and model disagree",
                "anthropic": ai_plan,
                "model_signal": model_signal,
                "model_confidence": round(float(model_conf), 4),
            }

        # 4) Execute through existing risk/order logic
        client = autotrader._client()
        position = autotrader._get_position(client)
        pos_side = position['side'] if position else None
        side = "Buy" if model_signal == "BUY" else "Sell"

        if (model_signal == "BUY" and pos_side == "Buy") or (model_signal == "SELL" and pos_side == "Sell"):
            autotrader._log(model_signal, model_conf, "HYBRID SKIP", f"already in {pos_side}")
            return {
                "status": "skip",
                "reason": "Already in same direction",
                "position_side": pos_side,
                "anthropic": ai_plan,
                "model_signal": model_signal,
                "model_confidence": round(float(model_conf), 4),
            }

        if pos_side and pos_side != side:
            close_side = "Buy" if pos_side == "Sell" else "Sell"
            cr = autotrader._place(client, close_side, float(position['size']), reduce_only=True)
            if cr.get('retCode') != 0:
                autotrader._log(model_signal, model_conf, "HYBRID FAILED",
                                f"close {pos_side}: retCode={cr.get('retCode')} {cr.get('retMsg')}")
                return JSONResponse(status_code=400, content={
                    "status": "failed",
                    "stage": "close_opposite",
                    "retCode": cr.get("retCode"),
                    "retMsg": cr.get("retMsg"),
                })
            await asyncio.sleep(1)

        sl, tp = autotrader._sl_tp_prices(client, side)
        trail = autotrader._trailing_distance(client)
        qty = autotrader._get_qty(client)
        rr = autotrader._place(client, side, qty, sl=sl, tp=tp, trailing=trail)
        if rr.get("retCode") != 0:
            autotrader._log(model_signal, model_conf, "HYBRID FAILED",
                            f"entry retCode={rr.get('retCode')} {rr.get('retMsg')}")
            return JSONResponse(status_code=400, content={
                "status": "failed",
                "stage": "entry",
                "retCode": rr.get("retCode"),
                "retMsg": rr.get("retMsg"),
                "anthropic": ai_plan,
                "model_signal": model_signal,
                "model_confidence": round(float(model_conf), 4),
            })

        autotrader._log(model_signal, model_conf, f"HYBRID OPENED {side.upper()} {qty} BTC",
                        f"anthropic={round(ai_plan['confidence']*100,1)}% model={round(model_conf*100,1)}%")
        return {
            "status": "ok",
            "symbol": symbol,
            "interval": interval,
            "side": side,
            "qty": qty,
            "orderId": rr.get("result", {}).get("orderId", ""),
            "anthropic": ai_plan,
            "model_signal": model_signal,
            "model_confidence": round(float(model_conf), 4),
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[Hybrid] error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── REST: train model ──────────────────────────────────────────────
@app.post("/api/train")
def train(symbol: str = "BTCUSDT", interval: str = "60", move_threshold: float = 0.0025,
          _auth=Depends(require_auth)):
    if interval not in ('1','3','5','15','30','60','120','240','360','720','D','W','M'):
        raise HTTPException(400, "Invalid interval")
    if move_threshold < 0.001 or move_threshold > 0.02:
        raise HTTPException(400, "move_threshold must be between 0.001 and 0.02")
    try:
        df = fetch_bybit_data(symbol=symbol, interval=interval, limit=1000,
                              api_key=API_KEY, api_secret=API_SECRET)
        X, y = preprocess_data(df, threshold=move_threshold)
        bot.train_model(X, y)
        bot.save_model()
        return {
            "status": "ok",
            "message": "MAHORAGA model trained and saved",
            "move_threshold": move_threshold
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── REST: chat ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    messages: List[dict]
    context:  str = ""

def _close_open_position() -> dict:
    client = get_bybit_client(API_KEY, API_SECRET)
    resp   = client.get_positions(category='linear', symbol='BTCUSDT')
    for p in resp['result']['list']:
        size = float(p.get('size', 0))
        if size > 0:
            close_side = 'Sell' if p['side'] == 'Buy' else 'Buy'
            r = client.place_order(
                category='linear', symbol='BTCUSDT',
                side=close_side, orderType='Market',
                qty=str(size), timeInForce='IOC',
                reduceOnly=True
            )
            log.info(f"[Chat] close_position: side={close_side} size={size} retCode={r.get('retCode')}")
            return r
    return {"retCode": -1, "retMsg": "No open position found"}

@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request, _auth=Depends(require_auth)):
    _require_anthropic()
    ip  = request.client.host
    now = time.time()

    if _check_rate(ip, 'chat', CHAT_RATE_LIMIT):
        return JSONResponse(status_code=429, content={"error": "Too many messages. Wait a moment."})

    def clean(text: str) -> str:
        return _STRIP_HTML.sub('', text).strip()[:MAX_MSG_LEN]

    messages = [
        {"role": m["role"], "content": clean(m.get("content", ""))}
        for m in req.messages[-10:]
        if m.get("role") in ("user", "assistant") and m.get("content", "").strip()
    ]
    if not messages:
        return JSONResponse(status_code=400, content={"error": "Empty message."})

    system = _SECURITY_SYSTEM
    if req.context:
        system += "\n\nLIVE ACCOUNT STATE (read-only reference):\n" + clean(req.context)

    log.info(f"[Chat] {ip} → {messages[-1]['content'][:80]}")

    try:
        ac   = anthropic.Anthropic()
        resp = ac.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system,
            tools=[_CLOSE_TOOL],
            messages=messages
        )

        if resp.stop_reason == "tool_use":
            tool_block = next((b for b in resp.content if b.type == "tool_use"), None)
            if tool_block and tool_block.name == "close_position":
                close_result = await asyncio.to_thread(_close_open_position)
                success      = close_result.get("retCode") == 0
                tool_result  = "Position closed successfully." if success else f"Failed: {close_result.get('retMsg', 'unknown error')}"

                followup = ac.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=128,
                    system=system,
                    tools=[_CLOSE_TOOL],
                    messages=messages + [
                        {"role": "assistant", "content": resp.content},
                        {"role": "user",      "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": tool_result}]}
                    ]
                )
                reply = next((b.text for b in followup.content if hasattr(b, "text")), tool_result)
                return {"reply": reply, "action": "position_closed", "success": success}

        reply = next((b.text for b in resp.content if hasattr(b, "text")), "No response.")
        return {"reply": reply}

    except Exception as e:
        log.error(f"[Chat] error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── REST: streaming chat ──────────────────────────────────────────
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, _auth=Depends(require_auth)):
    _require_anthropic()
    ip = request.client.host

    if _check_rate(ip, 'chat', CHAT_RATE_LIMIT):
        return JSONResponse(status_code=429, content={"error": "Too many messages. Wait a moment."})

    def clean(text: str) -> str:
        return _STRIP_HTML.sub('', text).strip()[:MAX_MSG_LEN]

    messages = [
        {"role": m["role"], "content": clean(m.get("content", ""))}
        for m in req.messages[-10:]
        if m.get("role") in ("user", "assistant") and m.get("content", "").strip()
    ]
    if not messages:
        return JSONResponse(status_code=400, content={"error": "Empty message."})

    system = _SECURITY_SYSTEM
    if req.context:
        system += "\n\nLIVE ACCOUNT STATE (read-only reference):\n" + clean(req.context)

    log.info(f"[Chat/Stream] {ip} → {messages[-1]['content'][:80]}")

    async def generate():
        try:
            import queue as _queue
            import threading as _threading
            ac = anthropic.Anthropic()
            tool_use_name = None
            tool_use_id   = None
            final_message = None
            text_q: _queue.Queue = _queue.Queue()

            def _stream_worker():
                nonlocal tool_use_name, tool_use_id, final_message
                try:
                    with ac.messages.stream(
                        model="claude-sonnet-4-6",
                        max_tokens=512,
                        system=system,
                        tools=[_CLOSE_TOOL],
                        messages=messages,
                    ) as stream:
                        for event in stream:
                            if event.type == "content_block_start":
                                blk = getattr(event, "content_block", None)
                                if blk and getattr(blk, "type", None) == "tool_use":
                                    tool_use_name = blk.name
                                    tool_use_id   = blk.id
                            elif event.type == "content_block_delta":
                                delta = getattr(event, "delta", None)
                                if delta and hasattr(delta, "text"):
                                    text_q.put(("text", delta.text))
                        final_message = stream.get_final_message()
                    text_q.put(("done", None))
                except Exception as ex:
                    text_q.put(("error", str(ex)))

            worker = _threading.Thread(target=_stream_worker, daemon=True)
            worker.start()

            # Immediately flush an SSE comment — this confirms the connection is
            # live to the browser and prevents nginx/Cloudflare from buffering
            # the response until the first real chunk arrives.
            yield ": connected\n\n"

            PING_EVERY = 5.0   # seconds — keeps the TCP connection alive through proxies
            last_ping  = asyncio.get_event_loop().time()

            while True:
                try:
                    kind, val = await asyncio.wait_for(
                        asyncio.to_thread(text_q.get),
                        timeout=1.5,
                    )
                except asyncio.TimeoutError:
                    # Queue is empty — send a keepalive ping comment so nginx/
                    # Cloudflare don't treat the idle connection as stalled.
                    now = asyncio.get_event_loop().time()
                    if now - last_ping >= PING_EVERY:
                        yield ": ping\n\n"
                        last_ping = now
                    # If the worker has finished and the queue is drained, exit.
                    if not worker.is_alive() and text_q.empty():
                        break
                    continue
                except Exception:
                    yield f"data: {json.dumps({'error': 'stream timeout'})}\n\n"
                    break

                if kind == "text":
                    yield f"data: {json.dumps({'text': val})}\n\n"
                    last_ping = asyncio.get_event_loop().time()
                elif kind == "error":
                    yield f"data: {json.dumps({'error': val})}\n\n"
                    break
                elif kind == "done":
                    break

            # Handle tool-use (e.g. close_position) after streaming completes
            if tool_use_name == "close_position" and tool_use_id and final_message:
                close_result = await asyncio.to_thread(_close_open_position)
                success      = close_result.get("retCode") == 0
                tool_result  = "Position closed successfully." if success else f"Failed: {close_result.get('retMsg', 'unknown error')}"

                followup = await asyncio.to_thread(
                    ac.messages.create,
                    model="claude-sonnet-4-6",
                    max_tokens=128,
                    system=system,
                    tools=[_CLOSE_TOOL],
                    messages=messages + [
                        {"role": "assistant", "content": final_message.content},
                        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": tool_result}]},
                    ],
                )
                followup_text = next((b.text for b in followup.content if hasattr(b, "text")), tool_result)
                yield f"data: {json.dumps({'text': followup_text, 'action': 'position_closed', 'success': success})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            log.error(f"[Chat/Stream] error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering":"no",          # nginx: disable proxy buffering
        },
    )





# ── REST: model status ────────────────────────────────────────────
@app.get("/api/model/status")
def model_status(_auth=Depends(require_auth)):
    return {
        "loaded": bot.model is not None,
        "has_scaler": bot.scaler is not None,
    }


# ── WebSocket: live feed ──────────────────────────────────────────
@app.websocket("/ws/price")
async def ws_price(ws: WebSocket):
    # Auth check before accepting
    token = ws.cookies.get(SESSION_COOKIE) or ws.query_params.get('token', '')
    if DASHBOARD_PASSWORD and not _valid_session(token):
        await ws.close(code=4001)
        log.warning(f"[WS] Rejected unauthenticated connection from {ws.client.host}")
        return

    await ws.accept()
    try:
        while True:
            try:
                client = get_bybit_client(API_KEY, API_SECRET)
                t      = await asyncio.to_thread(client.get_tickers, category='linear', symbol='BTCUSDT')
                tk     = t['result']['list'][0]
                price  = float(tk['lastPrice'])
                mark   = float(tk['markPrice'])
                prev24 = float(tk['prevPrice24h'])
                change = round((price - prev24) / prev24 * 100, 3)

                pr       = await asyncio.to_thread(client.get_positions, category='linear', symbol='BTCUSDT')
                pos_list = [p for p in pr['result']['list'] if float(p.get('size', 0)) > 0]
                position = None
                if pos_list:
                    p        = pos_list[0]
                    position = {
                        'side':          p['side'],
                        'size':          float(p['size']),
                        'entryPrice':    float(p['avgPrice']),
                        'markPrice':     mark,
                        'unrealisedPnl': float(p['unrealisedPnl']),
                        'liqPrice':      float(p['liqPrice']) if p.get('liqPrice') else 0,
                    }

                at = autotrader.last_signal
                await ws.send_json({
                    'price':      round(price, 2),
                    'markPrice':  round(mark, 2),
                    'change':     change,
                    'high24':     round(float(tk['highPrice24h']), 2),
                    'low24':      round(float(tk['lowPrice24h']), 2),
                    'position':   position,
                    'at_signal':  at,
                    'at_running': autotrader.running,
                    'at_next':    max(0, int((autotrader.next_check or 0) - datetime.now().timestamp())) if autotrader.running else 0,
                })
            except WebSocketDisconnect:
                break
            except Exception:
                pass
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        pass

def _setup_wizard(env_path: str):
    print("\n" + "=" * 52)
    print("   MAHORAGA — First Time Setup")
    print("=" * 52)
    print("\n  Paste each value and press Enter:\n")

    def _ask(label):
        while True:
            val = input(f"  {label}: ").strip()
            if val:
                return val
            print("  ⚠  Cannot be empty, try again.")

    bybit_key       = _ask("Bybit API Key     ")
    bybit_secret    = _ask("Bybit API Secret  ")
    anthropic_key   = _ask("Anthropic API Key ")
    dash_password   = _ask("Dashboard Password")

    with open(env_path, 'w') as f:
        f.write(f"BYBIT_API_KEY={bybit_key}\n")
        f.write(f"BYBIT_API_SECRET={bybit_secret}\n")
        f.write(f"ANTHROPIC_API_KEY={anthropic_key}\n")
        f.write(f"DASHBOARD_PASSWORD={dash_password}\n")

    print("\n  ✓ All set! Starting MAHORAGA...\n")

if __name__ == "__main__":
    import uvicorn

    # Run setup wizard on first launch (no external .env yet)
    if not os.path.exists(_EXT_ENV):
        _setup_wizard(_EXT_ENV)
        # Restart so all module-level env vars reload from the new .env
        os.execv(sys.executable, sys.argv)

    log.info(f"Starting MAHORAGA on {BIND_HOST}:{BIND_PORT}")
    if not DASHBOARD_PASSWORD:
        log.warning("⚠  Running without a password — set DASHBOARD_PASSWORD in .env")
    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT, reload=False)
