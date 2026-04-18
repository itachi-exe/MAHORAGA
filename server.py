import os, sys, json, asyncio, logging, re, time, secrets

# Resolve runtime directory — works both normally and inside a PyInstaller binary
# Bundled files (html, .env) live in _MEIPASS; model files live next to the executable
_BUNDLE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
_APP_DIR    = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) \
              else os.path.dirname(os.path.abspath(__file__))
from collections import defaultdict
from datetime import datetime, timezone, timedelta
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
from core_trading_system import (MAHORAGA, fetch_bybit_data, get_bybit_client,
                                 preprocess_data, virtual_trader_qty,
                                 perform_weighted_retraining)

try:
    from polymarket_trader import PolymarketTrader
    _POLY_IMPORT_OK = True
except Exception as _poly_import_err:
    PolymarketTrader = None  # type: ignore[assignment,misc]
    _POLY_IMPORT_OK  = False

# Load external .env (saved by wizard) first, fall back to bundled defaults
_EXT_ENV = os.path.join(_APP_DIR, '.env')
_BUN_ENV = os.path.join(_BUNDLE_DIR, '.env')
load_dotenv(dotenv_path=_BUN_ENV, override=False)   # bundled defaults
load_dotenv(dotenv_path=_EXT_ENV, override=True)    # client keys always win

API_KEY            = os.getenv('BYBIT_API_KEY', '')
API_SECRET         = os.getenv('BYBIT_API_SECRET', '')
ANTHROPIC_API_KEY  = os.getenv('ANTHROPIC_API_KEY', '')
DASHBOARD_PASSWORD = os.getenv('DASHBOARD_PASSWORD', os.getenv('CHAT_PASSWORD', '')).strip()
SETTINGS_PASSWORD  = os.getenv('SETTINGS_PASSWORD', '').strip()
BIND_HOST          = os.getenv('BIND_HOST', '127.0.0.1').strip()
BIND_PORT          = int(os.getenv('BIND_PORT', '8501'))

# ── Auto-retrain versioning ────────────────────────────────────────
AUTO_RETRAIN_EVERY       = 20    # closed trades between weighted retrains
AUTO_RETRAIN_MIN_TRADES  = 20    # minimum trades before first retrain
ROLLBACK_THRESHOLD       = 0.03  # roll back if val_accuracy drops by this

# ── MASTER AI CONTROL ─────────────────────────────────────────────
# Persisted to disk so server restarts honour the last manual switch state.
_MASTER_SWITCH_FILE = os.path.join(_APP_DIR, 'master_switch.json')

def _load_master_switch() -> bool:
    try:
        with open(_MASTER_SWITCH_FILE) as f:
            return bool(json.load(f).get('enabled', True))
    except Exception:
        return True  # default ON if file missing

def _save_master_switch(enabled: bool):
    with open(_MASTER_SWITCH_FILE, 'w') as f:
        json.dump({'enabled': enabled}, f)

AI_TRADING_ENABLED: bool = _load_master_switch()

# ── POLYMARKET SWITCH ──────────────────────────────────────────────
_POLY_SWITCH_FILE = os.path.join(_APP_DIR, 'poly_switch.json')

def _load_poly_switch() -> bool:
    try:
        with open(_POLY_SWITCH_FILE) as f:
            return bool(json.load(f).get('enabled', True))
    except Exception:
        return True  # default ON if file missing

def _save_poly_switch(enabled: bool):
    with open(_POLY_SWITCH_FILE, 'w') as f:
        json.dump({'enabled': enabled}, f)

POLY_TRADING_ENABLED: bool = _load_poly_switch()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('MAHORAGA')

if not DASHBOARD_PASSWORD:
    log.warning('⚠  DASHBOARD_PASSWORD is not set — dashboard is UNPROTECTED')

# ── STATS BASELINE (for dashboard reset) ──────────────────────────
_BASELINE_FILE = os.path.join(_APP_DIR, 'stats_baseline.json')

def _load_baseline() -> dict:
    try:
        with open(_BASELINE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_baseline(b: dict):
    with open(_BASELINE_FILE, 'w') as f:
        json.dump(b, f, indent=2)

_stats_baseline: dict = _load_baseline()

# ── RESPONSE CACHE (serves stale data when Bybit is unreachable) ──
_dashboard_cache: dict | None = None   # last successful /api/dashboard response
_signal_cache:    dict | None = None   # last successful /api/signal response

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
CHAT_RATE_LIMIT       = 15   # per minute per IP
ORDER_RATE_LIMIT      = 5    # per minute per IP
API_RATE_LIMIT        = 60   # per minute per IP for generic endpoints
PREDICTION_RATE_LIMIT = 30   # per minute per IP for /api/prediction
POLYMARKET_RATE_LIMIT = 20   # per minute per IP for /api/polymarket/*
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

# ── AUTOMATIC STARTUP SCHEDULER ───────────────────────────────────
async def _auto_startup_scheduler():
    """
    Automatically start the autotrader at 05:00 UTC each day.
    If the server starts after 05:00 UTC on the current day, _app_startup handles the
    immediate start. This scheduler then fires at the NEXT 05:00 UTC (following day).
    After each day's session ends (bot stopped by limits), this restarts it next morning.
    """
    while True:
        now        = datetime.now(timezone.utc)
        # Always target NEXT 05:00 UTC (today if before 05:00, else tomorrow)
        if now.hour < 5:
            next_start = now.replace(hour=5, minute=0, second=0, microsecond=0)
        else:
            next_start = (now + timedelta(days=1)).replace(
                hour=5, minute=0, second=0, microsecond=0
            )

        seconds_until_start = (next_start - now).total_seconds()
        log.info(
            f'[Scheduler] Next autotrader restart at '
            f'{next_start.strftime("%Y-%m-%d %H:%M UTC")} '
            f'({int(seconds_until_start/3600)}h {int((seconds_until_start%3600)/60)}m away)'
        )

        await asyncio.sleep(seconds_until_start)

        if AI_TRADING_ENABLED and bot.model and not autotrader.running:
            log.info('[Scheduler] 05:00 UTC — auto-starting autotrader')
            autotrader.start(auto_start=True)
        else:
            reason = []
            if not AI_TRADING_ENABLED: reason.append("AI trading disabled")
            if not bot.model:          reason.append("no model loaded")
            if autotrader.running:     reason.append("already running")
            log.info(f'[Scheduler] 05:00 UTC — skipping auto-start: {", ".join(reason)}')

# Start the scheduler as a background task
_startup_task = None

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

# ── Model versioning helpers ───────────────────────────────────────
_MODEL_VERSION_HISTORY_FILE = os.path.join(_APP_DIR, 'model_version_history.json')


def _load_version_history() -> list:
    try:
        with open(_MODEL_VERSION_HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save_version_history(history: list) -> None:
    try:
        with open(_MODEL_VERSION_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as exc:
        log.warning(f"[AutoRetrain] Failed to save version history: {exc}")


def get_next_model_version() -> int:
    """
    Scan _APP_DIR for MAHORAGA_model_v{N}.pkl files.
    If none exist, copy the live model to v0 first (preserves the original).
    Returns the next version number to write.
    """
    import shutil, glob as _glob
    pattern = os.path.join(_APP_DIR, 'MAHORAGA_model_v*.pkl')
    existing = _glob.glob(pattern)
    if not existing:
        v0_path = os.path.join(_APP_DIR, 'MAHORAGA_model_v0.pkl')
        v0_scaler = os.path.join(_APP_DIR, 'MAHORAGA_scaler_v0.pkl')
        live_model = os.path.join(_APP_DIR, 'MAHORAGA_model.pkl')
        live_scaler = os.path.join(_APP_DIR, 'MAHORAGA_scaler.pkl')
        if os.path.exists(live_model):
            shutil.copy2(live_model, v0_path)
            log.info("[AutoRetrain] Preserved original model → MAHORAGA_model_v0.pkl")
        if os.path.exists(live_scaler):
            shutil.copy2(live_scaler, v0_scaler)
        # Record v0 in history
        history = _load_version_history()
        if not any(e.get('version') == 0 for e in history):
            history.append({
                'version':          0,
                'timestamp':        datetime.now(timezone.utc).isoformat(),
                'val_accuracy':     bot.current_accuracy if 'bot' in dir() else 0.0,
                'trades_trained_on': 0,
                'trigger':          'cold_start',
                'is_live':          False,
            })
            _save_version_history(history)
        return 1
    nums = []
    for p in existing:
        base = os.path.basename(p)
        try:
            nums.append(int(base.replace('MAHORAGA_model_v', '').replace('.pkl', '')))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


def get_best_model_version() -> tuple:
    """Returns (version_number, val_accuracy) of the best logged version."""
    history = _load_version_history()
    if not history:
        return (0, 0.0)
    best = max(history, key=lambda e: e.get('val_accuracy', 0.0))
    return (best.get('version', 0), float(best.get('val_accuracy', 0.0)))


# ── AUTO TRADER ────────────────────────────────────────────────────
class AutoTrader:
    SYMBOL   = 'BTCUSDT'
    INTERVAL = '60'
    MIN_QTY  = 0.001

    # ── GHOST-IN-THE-MARKET RISK CONSTANTS (hardcoded, non-negotiable) ──
    # SL/TP are NOT hardcoded here — they are computed per-trade by
    # bot.compute_adaptive_params() and live in bot.last_risk_params.
    RISK_PER_TRADE       = 0.01    # 1%  — fraction of balance risked per chunk
    CONF_THRESHOLD       = 0.65    # 65% — minimum model confidence to trade
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
        # ── Auto-retrain counter ───────────────────────────────────────
        self.trades_since_retrain    = 0     # counts successful trade opens; retrains at 20
        self.auto_retrain_threshold  = AUTO_RETRAIN_EVERY
        self._retraining             = False # guard against concurrent retrains
        self._last_retrain_meta      = None  # dict: version, accuracy, delta, action, timestamp
        # ── Current trade tracking ─────────────────────────────────────
        self.current_trade           = None  # current open trade details
        self.auto_started            = False  # whether this session was auto-started
        self.wait_for_new_signal     = False  # wait for market clear before trading
        self._last_reset_time_ms     = 0     # timestamp for manual limit override
        # ── Adaptive risk engine state ─────────────────────────────────
        self.cycle_count             = 0     # incremented each run-loop cycle; drives ATR baseline refresh
        self.consecutive_wins        = 0     # updated from exchange PNL alongside consec_losses
        # ── Continual learning state ───────────────────────────────────
        self._last_retrain_time      = 0     # unix timestamp; enforces 6-hour retrain cooldown
        self._rolling_outcomes       = []    # last 20 trade outcomes ('win'/'loss') for rolling W/R
        self._last_df                = None  # most recent candle df; used by /api/ai-status

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
            position_val = risk_usdt / (bot.last_risk_params['stop_loss_pct'] / 100)
            qty = round(position_val / price, 3)
            return max(self.MIN_QTY, qty)
        except Exception:
            return self.MIN_QTY

    def _get_wallet(self, client):
        w = client.get_wallet_balance(accountType='UNIFIED', coin='USDT')
        coin_list = w.get('result', {}).get('list', [{}])
        coins = coin_list[0].get('coin', [{}]) if coin_list else [{}]
        return float((coins[0] if coins else {}).get('walletBalance', 0))

    def _load_trade_journal(self):
        jp = os.path.join(_APP_DIR, 'trade_journal.json')
        if os.path.exists(jp):
            try:
                with open(jp, 'r') as f: return json.load(f)
            except: pass
        return []

    def _save_trade_journal(self, data):
        jp = os.path.join(_APP_DIR, 'trade_journal.json')
        with open(jp, 'w') as f: json.dump(data, f)

    def _load_pending_journal(self):
        pp = os.path.join(_APP_DIR, 'pending_trades.json')
        if os.path.exists(pp):
            try:
                with open(pp, 'r') as f: return json.load(f)
            except: pass
        return []

    def _save_pending_journal(self, data):
        pp = os.path.join(_APP_DIR, 'pending_trades.json')
        with open(pp, 'w') as f: json.dump(data, f)

    def _sl_tp_prices(self, client, side):
        t     = client.get_tickers(category='linear', symbol=self.SYMBOL)
        price = float(t['result']['list'][0]['markPrice'])
        sl_dist = price * (bot.last_risk_params['stop_loss_pct'] / 100)
        tp_dist = price * (bot.last_risk_params['take_profit_pct'] / 100)
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
                
                # Filter out trades from before a manual limit reset
                valid_closed_trades = [t for t in closed_trades if int(t.get('updatedTime', t.get('createdTime', 0))) >= self._last_reset_time_ms]

                self.trades_today = len(valid_closed_trades) + self.current_position_entries
                # Consecutive losses / wins — list is newest-first; streaks are mutually exclusive
                l_count = 0
                w_count = 0
                for trade in valid_closed_trades:
                    pnl = float(trade.get('closedPnl', 0))
                    if pnl < 0 and w_count == 0:
                        l_count += 1
                    elif pnl > 0 and l_count == 0:
                        w_count += 1
                    else:
                        break
                self.consec_losses    = l_count
                self.consecutive_wins = w_count
            except Exception as inner_e:
                log.warning(f"Failed to fetch closed PNL for limits: {inner_e}")

            if self.trades_today >= self.MAX_TRADES_PER_DAY:
                return True, f"Max {self.MAX_TRADES_PER_DAY} trades/day reached."
            if self.consec_losses >= self.MAX_CONSEC_LOSSES:
                return True, f"{self.MAX_CONSEC_LOSSES} consecutive losses reached."

            return False, ""
        except Exception as e:
            log.error(f"Error checking daily limits: {e}")
            # HIGH-3: fail CLOSED — never continue trading through an API error
            return True, f"Safety halt: could not verify daily limits ({type(e).__name__}): {e}"

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
            self.cycle_count += 1
            # Check if AI trading is enabled
            if not AI_TRADING_ENABLED:
                self._log('SYSTEM', 0, 'STOPPED', 'AI trading disabled by master switch')
                self.stop()
                break
                
            # First decision cycle uses the initial delay set during start(), then normal cadence.
            cycle_delay = self._initial_delay if first_cycle else self.check_secs
            self.next_check = datetime.now().timestamp() + cycle_delay
            try:
                client = self._client()

                position = self._get_position(client)
                if position:
                    try:
                        side  = position['side']
                        sl, tp = self._sl_tp_prices(client, side)
                        trail  = self._trailing_distance(client)
                        
                        # Active Adjustment: Always ensure Trailing Stop is set to 1% (or intended PCT)
                        # We also ensure Sl/Tp are present.
                        client.set_trading_stop(
                            category='linear', symbol=self.SYMBOL,
                            stopLoss=str(sl), takeProfit=str(tp),
                            trailingStop=trail,
                            slTriggerBy='MarkPrice', tpTriggerBy='MarkPrice',
                            positionIdx=0
                        )
                        
                        # Recover current_trade for frontend if it's missing (e.g. after restart)
                        if self.current_trade is None:
                            # Re-establish direction lock
                            if self.current_day_direction is None:
                                self.current_day_direction = side.upper()
                            if self.current_position_entries == 0:
                                self.current_position_entries = 1
                            
                            self.current_trade = {
                                'side': side.upper(),
                                'confidence': 0.0, # Unknown as we restarted
                                'sl': float(position.get('stopLoss') or sl),
                                'tp': float(position.get('takeProfit') or tp),
                                'trail_pct': self.TRAILING_STOP_PCT,
                                'order_id': position.get('positionReqId', 'RESTORED')[:8],
                                'entry_time': datetime.now().isoformat(),
                                'chunks': self.current_position_entries,
                                'max_chunks': self.MAX_CHUNKS
                            }

                        # Log protection status occasionally to avoid log spam, 
                        # but ensure it runs every cycle for safety.
                        if not position.get('trailingStop') or not position.get('stopLoss'):
                            self._log('GUARD', 0, f'PROTECTED {side}',
                                      f'SL=${sl} TP=${tp} Trail={self.TRAILING_STOP_PCT}%')
                    except Exception as eg:
                        if "position size is zero" not in str(eg):
                            self._log('GUARD', 0, 'MONITORING ERROR', str(eg)[:80])

                limit_hit, limit_msg = self._check_daily_limits(client)
                if limit_hit:
                    _next_utc = (datetime.now(timezone.utc) + timedelta(days=1)).replace(
                        hour=5, minute=0, second=0, microsecond=0
                    )
                    self._log(
                        'RISK', 0, 'STOPPED FOR THE DAY',
                        f'{limit_msg} — will auto-restart at {_next_utc.strftime("%Y-%m-%d %H:%M UTC")}'
                    )
                    self.stop()
                    break

                df       = await asyncio.to_thread(
                                fetch_bybit_data, symbol=self.SYMBOL, interval=self.INTERVAL,
                                limit=300, api_key=API_KEY, api_secret=API_SECRET)
                self._last_df = df                                          # for /api/ai-status
                features = await asyncio.to_thread(bot.preprocess_single_bar, df.copy())
                if self.cycle_count % 100 == 0:
                    bot._atr_baseline = await asyncio.to_thread(bot.compute_atr_baseline, df)
                signal, confidence = await asyncio.to_thread(bot.predict, features, self.CONF_THRESHOLD)

                # 2.3 — Regime-aware confidence adjustment
                _regime = await asyncio.to_thread(bot.detect_regime, df)
                _regime_mult = {'trending': 1.10, 'volatile': 0.85, 'ranging': 1.0}
                confidence = max(0.01, min(0.99, confidence * _regime_mult.get(_regime, 1.0)))
                self.last_signal = {
                    'signal':     signal,
                    'confidence': round(float(confidence), 4),
                    'time':       datetime.now().strftime('%H:%M:%S'),
                }

                position = self._get_position(client)
                pos_side = position['side'] if position else None
                
                # If we had a trade in memory but position is now gone, it hit SL/TP
                if not pos_side and self.current_trade is not None:
                    self._log('SYSTEM', 0, 'POSITION_CLOSED', 'Position hit SL/TP or was manually closed')
                    self.current_trade = None
                    self.current_position_entries = 0
                    
                    try:
                        # Fetch the newest closed PNL
                        cpr = client.get_closed_pnl(category='linear', symbol=self.SYMBOL, limit=1)
                        clist = cpr.get('result', {}).get('list', [])
                        if clist:
                            last_pnl = float(clist[0].get('closedPnl', 0))
                            outcome = 'win' if last_pnl > 0 else 'loss'

                            # 2.1 — rolling win rate tracking (last 20 trades)
                            self._rolling_outcomes.append(outcome)
                            self._rolling_outcomes = self._rolling_outcomes[-20:]

                            pending = self._load_pending_journal()
                            journal = self._load_trade_journal()
                            for tp_item in pending:
                                tp_item['outcome'] = outcome
                                tp_item['pnl'] = last_pnl
                                journal.append(tp_item)
                            self._save_trade_journal(journal[-100:])  # keep last 100 to avoid bloat
                            self._save_pending_journal([]) # clear pending

                            # 2.1 — performance-degradation retrain trigger
                            if (len(self._rolling_outcomes) >= 20
                                    and self.trades_since_retrain >= 20
                                    and not self._retraining):
                                _rwr = self._rolling_outcomes.count('win') / len(self._rolling_outcomes)
                                if _rwr < 0.45:
                                    log.info(f"[AutoRetrain] Rolling win rate {_rwr:.1%} below threshold — triggering retrain")
                                    asyncio.create_task(self._auto_retrain())
                    except Exception as e:
                        log.error(f"[AutoTrader] Journaling closed trade failed: {e}")

                # ── GHOST: Confidence filter ──────────────────────────
                # Use higher confidence threshold for automatic startups to be more selective
                effective_threshold = 0.75 if self.auto_started else self.CONF_THRESHOLD
                
                # ── GHOST: Wait for new signal on startup ───────────
                if self.wait_for_new_signal and not pos_side:
                    if signal == 'HOLD':
                        self.wait_for_new_signal = False
                    else:
                        self._log(signal, confidence, 'IGNORED (STARTUP)', 'Waiting for HOLD signal to avoid jumping mid-trend')
                        first_cycle = False
                        await asyncio.sleep(cycle_delay)
                        continue

                if confidence < effective_threshold or signal == 'HOLD':
                    reason = f"signal={signal}, conf={round(confidence*100,1)}%, threshold={round(effective_threshold*100,1)}%"
                    if signal == 'HOLD':
                        reason += " (model says HOLD)"
                    else:
                        reason += f" (auto-start mode)" if self.auto_started else ""
                        reason += " (below threshold)"
                    self._log(signal, confidence, 'IGNORED', reason)

                # ── UTC Window Filter (No new entries 00:00-05:00 UTC) ──
                elif datetime.now(timezone.utc).hour < 5:
                    self._log(signal, confidence, 'IGNORED (TIME)', 
                              f'Trading blocked until 05:00 UTC (Current UTC Hour: {datetime.now(timezone.utc).hour})')
                    await asyncio.sleep(self.check_secs)
                    continue

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
                            self.current_trade            = None  # Clear current trade when position closed
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
                        _cooloff_secs = bot.last_risk_params['cooloff_hours'] * 3600
                        if elapsed < _cooloff_secs:
                            remaining = int(_cooloff_secs - elapsed)
                            h, m = remaining // 3600, (remaining % 3600) // 60
                            self._log(signal, confidence, 'IGNORED',
                                      f'STOP: Cooloff period active — {h}h {m}m remaining '
                                      f'(adaptive cooloff={bot.last_risk_params["cooloff_hours"]:.1f}h)')
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
                                model='claude-haiku-4-5-20251001',
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

                    # ── Regime gate ───────────────────────────────────
                    try:
                        from market_regime import get_regime_verdict as _get_regime
                        _regime_dir = "UP" if target_side == "Buy" else "DOWN"
                        _rv = _get_regime(_regime_dir)
                        if not _rv["approved"]:
                            self._log(signal, confidence, "IGNORED",
                                      f"[MAHORAGA] Trade blocked — {_rv['reason']}")
                            await asyncio.sleep(self.check_secs)
                            continue
                        log.info(
                            f"[MAHORAGA] Regime cleared — "
                            f"{_rv['regime_15m']} / {_rv['regime_4h']} "
                            f"strength={_rv['trend_strength']} placing order"
                        )
                    except Exception as _re:
                        log.warning(f"[MAHORAGA] Regime gate error: {_re} — proceeding")

                    # ── Execute trade ─────────────────────────────────
                    # Compute adaptive SL/TP/cooloff for this specific trade
                    _atr_pct = 0.015
                    try:
                        _atr_raw = float(df['atr'].iloc[-2]) if 'atr' in df.columns \
                                   else float(df['close'].iloc[-2]) * 0.015
                        _atr_pct = _atr_raw / float(df['close'].iloc[-2])
                    except Exception:
                        pass
                    context = {
                        'atr_pct':            _atr_pct,
                        'atr_baseline':       bot._atr_baseline,
                        'consecutive_wins':   self.consecutive_wins,
                        'consecutive_losses': self.consec_losses,
                        'confidence':         confidence,
                        'hour_utc':           datetime.now(timezone.utc).hour,
                    }
                    risk_params = bot.compute_adaptive_params(context)
                    bot.last_risk_params = risk_params
                    log.info(
                        f"[AdaptiveRisk] SL={risk_params['stop_loss_pct']:.3f}% "
                        f"TP={risk_params['take_profit_pct']:.3f}% "
                        f"Cooloff={risk_params['cooloff_hours']:.1f}h "
                        f"R:R={risk_params['r_ratio']} "
                        f"VolRatio={risk_params['vol_ratio']}"
                    )
                    # SL/TP prices using adaptive percentages (replaces _sl_tp_prices)
                    t_now   = client.get_tickers(category='linear', symbol=self.SYMBOL)
                    price   = float(t_now['result']['list'][0]['markPrice'])
                    sl_dist = price * (risk_params['stop_loss_pct']  / 100)
                    tp_dist = price * (risk_params['take_profit_pct'] / 100)
                    if target_side == 'Buy':
                        sl = round(price - sl_dist, 2)
                        tp = round(price + tp_dist, 2)
                    else:
                        sl = round(price + sl_dist, 2)
                        tp = round(price - tp_dist, 2)
                    trail = self._trailing_distance(client)
                    # Qty using adaptive SL for correct position sizing (replaces _get_qty)
                    try:
                        w_bal        = client.get_wallet_balance(accountType='UNIFIED', coin='USDT')
                        bal          = float(w_bal['result']['list'][0]['coin'][0]['walletBalance'])
                        risk_usdt    = bal * self.RISK_PER_TRADE
                        position_val = risk_usdt / (risk_params['stop_loss_pct'] / 100)
                        qty          = max(self.MIN_QTY, round(position_val / price, 3))
                    except Exception:
                        qty = self._get_qty(client)
                    r      = self._place(client, target_side, qty, sl=sl, tp=tp, trailing=trail)
                    if r.get('retCode') == 0:
                        self.current_position_entries += 1
                        self.last_trade_time = datetime.now()   # start 5hr cooloff
                        self.trades_since_retrain += 1
                        self.trades_today += 1  # Increment daily trade counter
                        # Store current trade details
                        self.current_trade = {
                            'side': target_side.upper(),
                            'confidence': confidence,
                            'sl': sl,
                            'tp': tp,
                            'trail_pct': self.TRAILING_STOP_PCT,
                            'order_id': r["result"].get("orderId","")[:8],
                            'entry_time': datetime.now().isoformat(),
                            'chunks': self.current_position_entries,
                            'max_chunks': self.MAX_CHUNKS
                        }
                        
                        try:
                            # Capture Reinforcement Learning features
                            fd = features.to_dict('records')[0] if not features.empty else {}
                            pending = self._load_pending_journal()
                            pending.append({
                                "features_at_entry": fd,
                                "label_at_entry": 2 if signal == 'BUY' else 0,
                                "time": datetime.now().isoformat()
                            })
                            self._save_pending_journal(pending)
                        except Exception as je:
                            log.error(f"[AutoTrader] Pending journal save failed: {je}")

                        self._log(signal, confidence,
                                  f'OPENED {target_side.upper()} {qty} BTC (chunk {self.current_position_entries}/{self.MAX_CHUNKS})',
                                  f'SL=${sl} TP=${tp} Trail={self.TRAILING_STOP_PCT}% | {r["result"].get("orderId","")[:8]}')
                        # ── Auto-retrain trigger ───────────────────────
                        if (self.trades_since_retrain >= self.auto_retrain_threshold
                                and not self._retraining):
                            asyncio.create_task(self._auto_retrain())
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

    async def _auto_retrain(self):
        """
        Background weighted-retrain with model versioning.
        Never blocks the trading loop — runs perform_weighted_retraining()
        in a thread executor. Original model preserved as v0 forever.
        6-hour cooldown between retrains.
        """
        import shutil as _shutil
        if not AI_TRADING_ENABLED:
            return
        # 6-hour cooldown gate
        if time.time() - self._last_retrain_time < 21600:
            remaining_h = (21600 - (time.time() - self._last_retrain_time)) / 3600
            self._log('SYSTEM', 0, 'AUTO-RETRAIN',
                      f'Skipped — cooldown active ({remaining_h:.1f}h remaining)')
            return

        self._retraining = True
        prev_acc = bot.current_accuracy
        self._log('SYSTEM', 0, 'AUTO-RETRAIN',
                  f'Triggered after {self.trades_since_retrain} trades '
                  f'(prev_accuracy={prev_acc:.4f}) — starting…')
        try:
            # ── Fetch base dataset ─────────────────────────────────────
            df = await asyncio.to_thread(
                fetch_bybit_data, symbol=self.SYMBOL, interval=self.INTERVAL,
                limit=1000, api_key=API_KEY, api_secret=API_SECRET
            )
            X_base, y_base = preprocess_data(df, threshold=0.0025)
            journal = await asyncio.to_thread(autotrader._load_trade_journal)
            self._log('SYSTEM', 0, 'AUTO-RETRAIN',
                      f'{len(y_base)} base samples, {len(journal)} journal trades — training…')

            # ── Determine next version; ensure v0 is preserved ────────
            next_ver = get_next_model_version()

            # ── Run weighted retrain in executor — never blocks loop ───
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                perform_weighted_retraining,
                bot, preprocess_data, journal, X_base, y_base,
            )

            new_acc = bot.current_accuracy
            delta   = new_acc - prev_acc
            now_iso = datetime.now(timezone.utc).isoformat()

            live_model  = os.path.join(_APP_DIR, 'MAHORAGA_model.pkl')
            live_scaler = os.path.join(_APP_DIR, 'MAHORAGA_scaler.pkl')
            vn_model    = os.path.join(_APP_DIR, f'MAHORAGA_model_v{next_ver}.pkl')
            vn_scaler   = os.path.join(_APP_DIR, f'MAHORAGA_scaler_v{next_ver}.pkl')
            prev_vn     = next_ver - 1  # version to roll back to if needed

            # ── Versioning + rollback decision ─────────────────────────
            # perform_weighted_retraining() already rolls back on any drop;
            # new_acc >= prev_acc means it succeeded and saved to MAHORAGA_model.pkl
            if new_acc >= prev_acc:
                # Snapshot the improved model as vN
                if os.path.exists(live_model):
                    _shutil.copy2(live_model, vn_model)
                if os.path.exists(live_scaler):
                    _shutil.copy2(live_scaler, vn_scaler)

                # Extra outer check: if drop > ROLLBACK_THRESHOLD, revert anyway
                if delta < -ROLLBACK_THRESHOLD:
                    prev_model  = os.path.join(_APP_DIR, f'MAHORAGA_model_v{prev_vn}.pkl')
                    prev_scaler = os.path.join(_APP_DIR, f'MAHORAGA_scaler_v{prev_vn}.pkl')
                    if os.path.exists(prev_model):
                        _shutil.copy2(prev_model, live_model)
                        bot.load_model(path=live_model, scaler_path=live_scaler)
                        action = 'ROLLED_BACK'
                        log.warning(
                            f'[AutoRetrain] ROLLBACK — new model underperforms '
                            f'keeping v{prev_vn}'
                        )
                    else:
                        action = 'UPDATED'
                else:
                    action = 'UPDATED'
                    log.info(
                        f'[AutoRetrain] v{next_ver} saved — '
                        f'val_accuracy={new_acc:.4f} '
                        f'previous={prev_acc:.4f} '
                        f'delta={delta:+.4f}'
                    )
            else:
                # perform_weighted_retraining() rolled back internally — no new version
                action = 'ROLLED_BACK'
                log.warning(
                    f'[AutoRetrain] ROLLBACK — new model underperforms '
                    f'keeping v{prev_vn if prev_vn >= 0 else 0}'
                )

            # ── Update version history (append-only) ──────────────────
            history = _load_version_history()
            # Mark all previous entries as not live
            if action == 'UPDATED':
                for entry in history:
                    entry['is_live'] = False
            history.append({
                'version':          next_ver if action == 'UPDATED' else prev_vn,
                'timestamp':        now_iso,
                'val_accuracy':     round(new_acc, 6),
                'trades_trained_on': len(journal),
                'trigger':          'auto',
                'is_live':          action == 'UPDATED',
                'action':           action,
                'delta':            round(delta, 6),
            })
            _save_version_history(history)

            # ── Store metadata for /api/ai-status ─────────────────────
            best_ver, best_acc = get_best_model_version()
            cur_ver = next_ver if action == 'UPDATED' else prev_vn
            self._last_retrain_meta = {
                'timestamp':      now_iso,
                'version':        cur_ver,
                'accuracy':       round(new_acc, 4),
                'delta':          round(delta, 4),
                'action':         action,
                'best_version':   best_ver,
                'best_accuracy':  round(best_acc, 4),
            }

            self.trades_since_retrain = 0
            self._last_retrain_time   = time.time()
            self._log('SYSTEM', 0, 'AUTO-RETRAIN',
                      f'{action} — v{cur_ver} acc={new_acc:.4f} delta={delta:+.4f}')

        except Exception as exc:
            self._log('SYSTEM', 0, 'AUTO-RETRAIN-ERR', str(exc)[:200])
            log.exception('[AutoRetrain] Unhandled error — keeping current model')
        finally:
            self._retraining = False

    def start(self, auto_start=False):
        if not AI_TRADING_ENABLED:
            log.warning("[AutoTrader] Cannot start - AI trading is disabled by master switch")
            return
        if not bot.model:
            log.warning("[AutoTrader] Cannot start - no model loaded")
            return
        if not self.running:
            self.running         = True
            self._daily_loss_ref = None
            self.auto_started    = auto_start  # Track if this was an automatic start
            self.wait_for_new_signal = True    # Force it to wait for a signal transition before entering
            # Use longer initial delay for automatic startup to allow for better trade selection
            self._initial_delay = 300 if auto_start else self.first_delay_secs  # 5 minutes for auto-start
            self._log('SYSTEM', 0, 'STARTED' + (' (AUTO)' if auto_start else ''),
                      f'first check in {self._initial_delay}s, then every {self.check_secs}s')
            self.task            = asyncio.create_task(self.run())
            log.info(f'[AutoTrader] Started{" (auto)" if auto_start else ""}')

    def stop(self):
        self.running = False
        self.current_trade = None  # Clear current trade when stopping
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
            cooloff_remaining = max(0, int(bot.last_risk_params.get('cooloff_hours', 5) * 3600 - elapsed))
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
            'trades_since_retrain':  self.trades_since_retrain,
            'auto_retrain_threshold': self.auto_retrain_threshold,
            'retraining':            self._retraining,
            'current_trade':         self.current_trade,
            'auto_started':          self.auto_started,
            'log':                   self.trade_log[:20],
            # Hardcoded Ghost-algorithm constants (read-only)
            'risk_constants': {
                'risk_per_trade_pct':   self.RISK_PER_TRADE * 100,
                'conf_threshold_pct':   self.CONF_THRESHOLD * 100,
                'stop_loss_pct':        bot.last_risk_params['stop_loss_pct'],
                'take_profit_pct':      bot.last_risk_params['take_profit_pct'],
                'trailing_stop_pct':    self.TRAILING_STOP_PCT,
                'max_daily_loss_pct':   self.MAX_DAILY_LOSS_PCT,
                'max_daily_profit_pct': self.MAX_DAILY_PROFIT_PCT,
                'max_trades_per_day':   self.MAX_TRADES_PER_DAY,
                'max_consec_losses':    self.MAX_CONSEC_LOSSES,
                'max_chunks':           self.MAX_CHUNKS,
            }
        }


# ── VIRTUAL TRADER (PAPER TRADING) ─────────────────────────────────
class VirtualTrader:
    SYMBOL = 'BTCUSDT'
    INTERVAL = '60'
    FEE_PCT = 0.0006  # 0.06% Bybit taker fee simulation

    def __init__(self):
        self.running = False
        self.task = None
        self._state_file = os.path.join(_APP_DIR, 'paper_state.json')
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, 'r') as f:
                    return json.load(f)
            except: pass
        return {'balance': 10000.0, 'position': None}

    def _save_state(self):
        with open(self._state_file, 'w') as f:
            json.dump(self.state, f)

    def start(self):
        if not AI_TRADING_ENABLED:
            log.warning("[VirtualTrader] Cannot start — system disabled by master switch")
            return
        if self.running: return
        self.running = True
        self.task = asyncio.create_task(self.run())
        log.info("[VirtualTrader] Simulator started.")

    def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
        log.info("[VirtualTrader] Simulator stopped.")

    def update_balance(self, amount: float):
        self.state['balance'] += amount
        self._save_state()

    def status(self):
        pos = self.state.get('position')
        unrealized_pnl = None
        if pos:
            try:
                client = get_bybit_client(API_KEY, API_SECRET)
                t = client.get_tickers(category='linear', symbol=self.SYMBOL)
                mark = float(t['result']['list'][0]['markPrice'])
                ep = pos['entry_price']
                q  = pos['qty']
                raw = (mark - ep) * q if pos['side'] == 'Buy' else (ep - mark) * q
                fee = (ep * q * self.FEE_PCT) + (mark * q * self.FEE_PCT)
                unrealized_pnl = round(raw - fee, 4)
            except Exception:
                pass
        return {
            'running':        self.running,
            'balance':        self.state['balance'],
            'position':       pos,
            'unrealized_pnl': unrealized_pnl,
        }

    async def run(self):
        while self.running:
            try:
                client = get_bybit_client(API_KEY, API_SECRET)
                t = client.get_tickers(category='linear', symbol=self.SYMBOL)
                current_price = float(t['result']['list'][0]['markPrice'])
                
                pos = self.state.get('position')

                if pos:
                    # Check SL/TP
                    hit_sl = False
                    hit_tp = False
                    if pos['side'] == 'Buy':
                        if current_price <= pos['sl']: hit_sl = True
                        if current_price >= pos['tp']: hit_tp = True
                    else:
                        if current_price >= pos['sl']: hit_sl = True
                        if current_price <= pos['tp']: hit_tp = True

                    if hit_sl or hit_tp:
                        entry_price = pos['entry_price']
                        qty         = pos['qty']
                        val         = entry_price * qty

                        if pos['side'] == 'Buy':
                            raw_pnl = (current_price - entry_price) * qty
                        else:
                            raw_pnl = (entry_price - current_price) * qty

                        fee     = (val * self.FEE_PCT) + (current_price * qty * self.FEE_PCT)
                        net_pnl = raw_pnl - fee

                        self.state['balance'] += net_pnl
                        self.state['position'] = None
                        self._save_state()

                        jp = os.path.join(_APP_DIR, 'trade_journal.json')
                        journal = []
                        if os.path.exists(jp):
                            try:
                                with open(jp, 'r') as f: journal = json.load(f)
                            except: pass

                        journal.append({
                            # ── user-facing trade fields ──────────────────
                            "side":           pos['side'],
                            "entry_price":    round(entry_price, 2),
                            "exit_price":     round(current_price, 2),
                            "qty":            qty,
                            "sl":             round(pos['sl'], 2),
                            "tp":             round(pos['tp'], 2),
                            "hit":            "TP" if hit_tp else "SL",
                            "outcome":        "win" if net_pnl > 0 else "loss",
                            "pnl":            round(net_pnl, 4),
                            "balance_after":  round(self.state['balance'], 4),
                            "entry_time":     pos.get('entry_time', ''),
                            "time":           datetime.now().isoformat(),
                            "virtual":        True,
                            # ── RL training fields ────────────────────────
                            "features_at_entry": pos.get('features', {}),
                            "label_at_entry":    pos.get('label', 1),
                        })
                        with open(jp, 'w') as f: json.dump(journal[-200:], f)

                        log.info(f"[VirtualTrader] CLOSED {pos['side']} @ {current_price:.2f} "
                                 f"({'TP' if hit_tp else 'SL'}) net_pnl={net_pnl:.2f} "
                                 f"balance={self.state['balance']:.2f}")

                # Fresh read of state in case position closed
                if not self.state.get('position') and AI_TRADING_ENABLED and bot.model:
                    df       = await asyncio.to_thread(fetch_bybit_data, symbol=self.SYMBOL,
                                                       interval=self.INTERVAL, limit=300,
                                                       api_key=API_KEY, api_secret=API_SECRET)
                    features = await asyncio.to_thread(bot.preprocess_single_bar, df.copy())
                    if features.empty:
                        log.debug("[VirtualTrader] preprocess_single_bar returned empty — skipping cycle")
                    else:
                        signal, confidence = await asyncio.to_thread(bot.predict, features, 0.60)

                        if signal in ['BUY', 'SELL'] and confidence >= 0.60:
                            side    = 'Buy' if signal == 'BUY' else 'Sell'
                            qty     = virtual_trader_qty(self.state['balance'], current_price)
                            sl_dist = current_price * 0.01
                            tp_dist = current_price * 0.03
                            sl      = current_price - sl_dist if side == 'Buy' else current_price + sl_dist
                            tp      = current_price + tp_dist if side == 'Buy' else current_price - tp_dist
                            fd      = features.to_dict('records')[0] if not features.empty else {}

                            self.state['position'] = {
                                'side':        side,
                                'entry_price': current_price,
                                'qty':         qty,
                                'sl':          round(sl, 2),
                                'tp':          round(tp, 2),
                                'confidence':  round(confidence, 4),
                                'features':    fd,
                                'label':       2 if signal == 'BUY' else 0,
                                'entry_time':  datetime.now().isoformat(),
                            }
                            self._save_state()
                            log.info(f"[VirtualTrader] OPENED {side} {qty:.3f} BTC @ {current_price:.2f} "
                                     f"SL={sl:.2f} TP={tp:.2f} conf={confidence:.2%}")
                        else:
                            log.debug(f"[VirtualTrader] Signal={signal} conf={confidence:.2%} — no entry")

            except Exception as e:
                log.warning(f"[VirtualTrader] Cycle error: {e}")

            await asyncio.sleep(60)

autotrader = AutoTrader()
virtual_trader = VirtualTrader()

# ── POLYMARKET TRADER ─────────────────────────────────────────────
poly_trader: "PolymarketTrader | None" = None
if _POLY_IMPORT_OK:
    try:
        poly_trader = PolymarketTrader()
    except Exception as _poly_init_err:
        log.warning(f"[Polymarket] Failed to initialize: {_poly_init_err}. Polymarket trading disabled.")

def _get_mahoraga_direction() -> str | None:
    """
    Return the MAHORAGA Bybit signal as a Polymarket direction ("UP" / "DOWN" / None).

    Uses autotrader.last_signal (the most recent MLP signal evaluated in the Bybit loop).
    Maps: BUY → "UP",  SELL → "DOWN",  anything else → None (neutral / no signal).
    Returns None when there is no recent signal so Polymarket can decide independently.
    """
    sig = getattr(autotrader, "last_signal", None)
    if not sig or not isinstance(sig, dict):
        return None
    raw = str(sig.get("signal", "")).upper()
    if raw == "BUY":
        return "UP"
    if raw == "SELL":
        return "DOWN"
    return None


async def _poly_autonomous_loop():
    """
    Autonomous Polymarket betting loop — runs every 60s independently of the dashboard.

    Signal alignment: Polymarket only bets when its MLP direction matches the live
    MAHORAGA Bybit signal (last_signal). If they conflict the bet is skipped — this
    prevents betting against what the main model is actually trading.

    Deliberately does NOT gate on AI_TRADING_ENABLED — Polymarket has its own switch.
    """
    await asyncio.sleep(30)  # brief delay to let poly_trader.setup() finish first
    log.info("[PolyAuto] Autonomous betting loop started")
    while True:
        try:
            if POLY_TRADING_ENABLED and poly_trader is not None:
                if bot.model is None:
                    log.debug("[PolyAuto] Model not loaded yet — skipping tick")
                else:
                    data = await bot.get_prediction_snapshot()
                    direction  = data.get("direction", "NEUTRAL")
                    confidence = float(data.get("confidence", 0))

                    # ── MAHORAGA signal alignment ──────────────────────
                    mahoraga_dir = _get_mahoraga_direction()
                    if mahoraga_dir and mahoraga_dir != direction:
                        log.info(
                            f"[PolyAuto] MAHORAGA conflict — MLP says {direction}, "
                            f"Bybit signals {mahoraga_dir} → skip this candle"
                        )
                    elif direction not in ("NEUTRAL", "OFFLINE"):
                        if mahoraga_dir:
                            log.info(
                                f"[PolyAuto] Signal: {direction} @ {confidence:.1f}% "
                                f"[MAHORAGA confirmed ✓] — attempting bet"
                            )
                        else:
                            log.info(
                                f"[PolyAuto] Signal: {direction} @ {confidence:.1f}% "
                                f"[no Bybit signal yet] — attempting bet"
                            )
                        asyncio.create_task(
                            poly_trader.place_bet(direction, confidence,
                                                  mahoraga_signal=mahoraga_dir)
                        )
                    else:
                        log.debug(
                            f"[PolyAuto] No bet: {direction} @ {confidence:.1f}%"
                        )

            # ── Paper trading (always runs when enabled, independent of real switch) ──
            if poly_trader is not None and poly_trader._paper_mode and bot.model is not None:
                try:
                    snap = await bot.get_prediction_snapshot()
                    pdir  = snap.get("direction", "NEUTRAL")
                    pconf = float(snap.get("confidence", 0))
                    if pdir not in ("NEUTRAL", "OFFLINE"):
                        asyncio.create_task(poly_trader.place_paper_bet(pdir, pconf))
                except Exception as _pe:
                    log.debug(f"[PaperAuto] tick error: {_pe}")

        except Exception as _loop_e:
            log.warning(f"[PolyAuto] Loop tick error (will retry in 60s): {_loop_e}")
        await asyncio.sleep(60)


# ── Background Bybit cache refresher ──────────────────────────────
async def _bybit_cache_refresh_loop():
    """
    Proactively refresh _dashboard_cache and _signal_cache every 30s.
    Runs entirely in executor threads so the event loop is never blocked.
    Endpoints always serve from cache — zero user-facing Bybit latency.
    """
    global _dashboard_cache, _signal_cache
    loop = asyncio.get_event_loop()
    # Give uvicorn a moment to finish booting before the first Bybit call
    await asyncio.sleep(5)
    while True:
        # ── Dashboard ────────────────────────────────────────────
        try:
            payload = await loop.run_in_executor(None, _build_dashboard_payload)
            _dashboard_cache = payload
            log.info("[CacheRefresh] Dashboard cache warmed")
        except Exception as _e:
            log.warning(f"[CacheRefresh] Dashboard failed (Bybit unreachable): {_e}")

        # ── Signal ───────────────────────────────────────────────
        if AI_TRADING_ENABLED and bot.model is not None:
            try:
                result = await loop.run_in_executor(None, _build_signal_payload)
                _signal_cache = result
                log.info(f"[CacheRefresh] Signal cache warmed: {result.get('signal')} @ {result.get('confidence',0)*100:.1f}%")
            except Exception as _e:
                log.warning(f"[CacheRefresh] Signal failed (Bybit unreachable): {_e}")

        await asyncio.sleep(30)


@app.on_event("startup")
async def _app_startup():
    """
    FastAPI startup event — runs regardless of how uvicorn is launched.

    Starts both the Polymarket trader and the Bybit auto-start scheduler here
    (not just in __main__) so they work whether the server is launched with
    `python3 server.py` or `uvicorn server:app`.
    """
    global _startup_task

    # ── Bybit autotrader scheduler ────────────────────────────────────
    if _startup_task is None or _startup_task.done():
        _startup_task = asyncio.create_task(_auto_startup_scheduler())
        log.info("[AutoTrader] Startup scheduler started")

    # ── Background Bybit cache refresher ─────────────────────────────
    asyncio.create_task(_bybit_cache_refresh_loop())
    log.info("[CacheRefresh] Background Bybit cache refresh started")

    # If the server (re)started after 05:00 UTC and autotrader is idle, start immediately
    # instead of waiting until tomorrow 05:00 UTC.
    _now_utc = datetime.now(timezone.utc)
    if (_now_utc.hour >= 5
            and AI_TRADING_ENABLED
            and bot.model is not None
            and not autotrader.running):
        log.info(
            f"[AutoTrader] Server started at {_now_utc.strftime('%H:%M')} UTC "
            f"(past 05:00 window) — auto-starting immediately"
        )
        autotrader.start(auto_start=True)

    # ── Polymarket trader ─────────────────────────────────────────────
    if poly_trader is not None:
        try:
            await poly_trader.setup()
            asyncio.create_task(poly_trader.run_loop())
            asyncio.create_task(_poly_autonomous_loop())
            log.info("[Polymarket] Trader + autonomous loop started.")
        except Exception as _e:
            log.warning(f"[Polymarket] Startup failed: {_e}")


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
        log.warning(f"[SettingsAuth] Rate limit hit from {ip}")
        return JSONResponse(status_code=429, content={"error": "Too many attempts."})
    
    if not SETTINGS_PASSWORD:
        log.error("[SettingsAuth] SETTINGS_PASSWORD is not set in .env! Rejecting all attempts.")
        return JSONResponse(status_code=401, content={"error": "Settings password not configured on server."})

    if not secrets.compare_digest(req.password.strip(), SETTINGS_PASSWORD):
        log.warning(f"[SettingsAuth] Wrong password from {ip}")
        await asyncio.sleep(1)
        return JSONResponse(status_code=401, content={"error": "Wrong settings password."})
    
    log.info(f"[SettingsAuth] Access granted to {ip}")
    return JSONResponse(content={"ok": True})

# ── AI Control Settings ───────────────────────────────────────────
class AISettingsRequest(BaseModel):
    enabled: bool

# ── Polymarket Switch ─────────────────────────────────────────────
class PolySettingsRequest(BaseModel):
    enabled: bool

class PaperConfigRequest(BaseModel):
    enabled: bool
    balance:  Optional[float] = None   # resets paper fund when provided
    bet_size: Optional[float] = None

@app.get("/api/settings/polymarket")
async def get_poly_settings(_auth=Depends(require_auth)):
    return {"poly_trading_enabled": POLY_TRADING_ENABLED}

@app.post("/api/settings/polymarket")
async def set_poly_settings(req: PolySettingsRequest, _auth=Depends(require_auth)):
    global POLY_TRADING_ENABLED
    POLY_TRADING_ENABLED = req.enabled
    _save_poly_switch(POLY_TRADING_ENABLED)
    log.info(f"[PolySwitch] Polymarket trading {'ENABLED' if POLY_TRADING_ENABLED else 'DISABLED'}")
    return {"poly_trading_enabled": POLY_TRADING_ENABLED}

@app.get("/api/settings/ai")
async def get_ai_settings(_auth=Depends(require_auth)):
    """Get current AI trading settings."""
    return {"ai_trading_enabled": AI_TRADING_ENABLED}

@app.post("/api/settings/ai")
async def set_ai_settings(req: AISettingsRequest, _auth=Depends(require_auth)):
    """Enable/disable all AI trading functionality."""
    global AI_TRADING_ENABLED
    old_state = AI_TRADING_ENABLED
    AI_TRADING_ENABLED = req.enabled
    
    _save_master_switch(AI_TRADING_ENABLED)

    if not AI_TRADING_ENABLED:
        if autotrader.running:
            autotrader.stop()
            log.info("[MasterSwitch] Autotrader stopped")
        if virtual_trader.running:
            virtual_trader.stop()
            log.info("[MasterSwitch] VirtualTrader stopped")

    log.info(f"[MasterSwitch] System {'ENABLED' if AI_TRADING_ENABLED else 'DISABLED'} "
             f"(was {'enabled' if old_state else 'disabled'})")

    return {"ai_trading_enabled": AI_TRADING_ENABLED}

# ── Dashboard HTML ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(_BUNDLE_DIR, "MAHORAGA_dashboard.html")
    with open(html_path, "r") as f:
        content = f.read()
    return HTMLResponse(content=content, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma":        "no-cache",
        "Expires":       "0",
    })

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
    global _signal_cache
    if not AI_TRADING_ENABLED:
        return {"signal": "OFFLINE", "confidence": 0, "model_loaded": False, "disabled": True}
    # Serve from cache if warm — background task keeps it fresh
    if _signal_cache is not None:
        return {**_signal_cache, "stale": False}
    # Cache cold: try live call
    try:
        result = _build_signal_payload(symbol, interval, confidence)
        _signal_cache = result
        return result
    except Exception as e:
        log.warning(f"[/api/signal] cold-cache Bybit error: {e}")
        return JSONResponse(status_code=503, content={
            "error": "Bybit unreachable — cache warming, retry in 30s",
            "stale": True, "bybit_offline": True
        })

# ── REST: balance ──────────────────────────────────────────────────
@app.get("/api/balance")
def balance(_auth=Depends(require_auth)):
    try:
        bal = bot.get_balance(API_KEY, API_SECRET)
        return {"usdt": round(bal, 2)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── Bybit data builder (used by endpoint + background refresh) ─────
def _build_dashboard_payload(symbol: str = "BTCUSDT") -> dict:
    """Fetch Bybit data and return the full dashboard dict. Raises on failure."""
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
    cpr       = client.get_closed_pnl(category='linear', symbol=symbol, limit=50)
    closed_trades = cpr.get('result', {}).get('list', [])

    def get_ts(c):
        return int(c.get('updatedTime', c.get('createdTime', 0)))

    today_closed = [c for c in closed_trades if get_ts(c) >= today_ms]
    today_pnl    = sum(float(c.get('closedPnl', 0)) for c in today_closed)

    bl             = _stats_baseline
    since_ms       = int(bl.get('since_time_ms', 0))
    cum_offset     = float(bl.get('cumRealisedPnl_offset', 0))
    today_offset   = float(bl.get('todayPnl_offset', 0))

    visible_closed     = [c for c in closed_trades if get_ts(c) > since_ms]
    trade_history = []
    for c in visible_closed[:20]:
        trade_history.append({
            'symbol':    c.get('symbol', ''),
            'side':      c.get('side', ''),
            'execPrice': float(c.get('avgEntryPrice', 0)),
            'execQty':   float(c.get('closedSize', 0)),
            'execTime':  get_ts(c),
            'closedPnl': float(c.get('closedPnl', 0))
        })

    raw_cum = float(coin.get('cumRealisedPnl', 0))

    active_chunks = 0
    if autotrader.running:
        active_chunks = autotrader.current_position_entries
    else:
        active_chunks = 1 if len(positions) > 0 else 0

    return {
        'walletBalance':  round(float(coin.get('walletBalance',  0)), 4),
        'equity':         round(float(coin.get('equity',         0)), 4),
        'unrealisedPnl':  round(float(coin.get('unrealisedPnl',  0)), 4),
        'cumRealisedPnl': round(raw_cum - cum_offset, 4),
        'todayPnl':       round(today_pnl - today_offset, 4),
        'todayTrades':    max(0, len(today_closed) + active_chunks),
        'totalTrades':    max(0, len(visible_closed) + active_chunks),
        'positions':      positions,
        'tradeHistory':   trade_history,
        'last_reset':     bl.get('reset_time'),
    }


def _build_signal_payload(symbol: str = "BTCUSDT", interval: str = "60",
                          confidence: float = 0.6) -> dict:
    """Fetch Bybit candles and return the signal dict. Raises on failure."""
    df = fetch_bybit_data(symbol=symbol, interval=interval, limit=300,
                          api_key=API_KEY, api_secret=API_SECRET)
    features = bot.preprocess_single_bar(df.copy())
    sig, conf = bot.predict(features, confidence)
    return {"signal": sig, "confidence": round(float(conf), 4),
            "model_loaded": bot.model is not None}


# ── REST: full dashboard data ──────────────────────────────────────
@app.get("/api/dashboard")
def dashboard_data(symbol: str = "BTCUSDT", _auth=Depends(require_auth)):
    global _dashboard_cache
    # Always serve from cache if warm — background task keeps it fresh
    if _dashboard_cache is not None:
        return {**_dashboard_cache, "autotrader": autotrader.status()}
    # Cache cold (first boot or first request after restart): try live call
    try:
        result = _build_dashboard_payload(symbol)
        result['autotrader'] = autotrader.status()
        _dashboard_cache = {k: v for k, v in result.items() if k != 'autotrader'}
        return result
    except Exception as e:
        log.warning(f"[/api/dashboard] cold-cache Bybit error: {e}")
        return JSONResponse(status_code=503, content={
            "error": "Bybit unreachable — cache warming, retry in 30s",
            "stale": True, "bybit_offline": True
        })

# ── REST: reset dashboard stats ────────────────────────────────────
@app.post("/api/stats/reset")
def stats_reset(symbol: str = "BTCUSDT", _auth=Depends(require_auth)):
    """
    Snapshots current Bybit stats as the new baseline.
    All display counters (P&L, trade count, history) will show 0 / empty after this.
    Only the in-memory autotrader log is cleared — trade_journal.json is intentionally
    preserved so RL training data is never destroyed by a stats reset (LOW-4).
    """
    global _stats_baseline
    try:
        client      = get_bybit_client(API_KEY, API_SECRET)
        now_utc     = datetime.now(timezone.utc)
        today_ms    = int(now_utc.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        since_ms    = int(now_utc.timestamp() * 1000)   # hide everything up to right now

        # Fetch current cumulative P&L
        w           = client.get_wallet_balance(accountType='UNIFIED', coin='USDT')
        coin_list   = w.get('result', {}).get('list', [{}])
        coins       = coin_list[0].get('coin', [{}]) if coin_list else [{}]
        coin        = coins[0] if coins else {}
        cum_pnl     = float(coin.get('cumRealisedPnl', 0))

        # Fetch today's execution count
        er          = client.get_executions(category='linear', limit=200)
        execs       = er['result']['list']
        today_ex    = [e for e in execs if int(e.get('execTime', 0)) >= today_ms]
        today_pnl   = sum(float(e.get('closedPnl', 0)) - float(e.get('execFee', 0)) for e in today_ex)

        _stats_baseline = {
            'reset_time':           now_utc.strftime('%Y-%m-%d %H:%M UTC'),
            'since_time_ms':        since_ms,
            'cumRealisedPnl_offset': cum_pnl,
            'totalTrades_offset':   len(execs),
            'todayPnl_offset':      today_pnl,
            'todayTrades_offset':   len(today_ex),
        }
        _save_baseline(_stats_baseline)

        # Clear in-memory autotrader log only — RL journal is never touched by
        # a stats reset (LOW-4: decoupled so RL training data is preserved).
        autotrader.trade_log = []

        log.info('[Stats] Dashboard reset by user')
        return {'ok': True, 'reset_time': _stats_baseline['reset_time']}
    except Exception as e:
        log.error(f'[Stats reset] {e}')
        return JSONResponse(status_code=500, content={'error': str(e)})

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
    if not AI_TRADING_ENABLED:
        return JSONResponse(status_code=400, content={"error": "AI trading is disabled by master switch"})
    if not bot.model:
        return JSONResponse(status_code=400, content={"error": "No model loaded. Train first."})
    autotrader.start()
    return {"status": "started"}

@app.post("/api/autotrader/stop")
async def at_stop(_auth=Depends(require_auth)):
    autotrader.stop()
    return {"status": "stopped"}

@app.post("/api/autotrader/reset-limits")
async def at_reset_limits(_auth=Depends(require_auth)):
    try:
        autotrader._last_reset_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # Reset the daily start balance so profit limits track from the current state
        try:
            client = autotrader._client()
            bal = autotrader._get_wallet(client)
            autotrader.daily_start_balance = bal
        except Exception:
            pass # fallback or update on next cycle
        return {"status": "ok", "message": "Autotrader active limits logic reset. Daily loss & consecutive loss rules are clear."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/autotrader/status")
async def at_status(_auth=Depends(require_auth)):
    return autotrader.status()


# ── REST: prediction snapshot ──────────────────────────────────────
@app.get("/api/prediction")
async def prediction_snapshot(request: Request, _auth=Depends(require_auth)):
    """
    Read-only live prediction snapshot.
    Returns direction, confidence, ob_imbalance, funding_rate, and
    seconds until the next 15-min candle close.
    No orders are placed by this endpoint.
    NOTE: NOT gated on AI_TRADING_ENABLED — Polymarket page needs this signal
    regardless of whether the Bybit autotrader is on.
    """
    if bot.model is None:
        return JSONResponse(content={
            "direction": "NEUTRAL", "confidence": 0, "disabled": False,
            "candle_close_countdown_s": int(900 - (time.time() % 900)),
            "signal_raw": "HOLD", "ob_imbalance": 0, "funding_rate": 0,
            "adaptive_sl": 1.0, "adaptive_tp": 3.0, "r_ratio": 3.0, "adaptive_cooloff": 5.0,
        })
    ip = request.client.host
    if _check_rate(ip, 'prediction', PREDICTION_RATE_LIMIT):
        return JSONResponse(status_code=429,
                            content={"error": "Too many prediction requests."})
    try:
        data = await bot.get_prediction_snapshot()
        # ── Polymarket auto-bet (fire-and-forget, one per 15-min candle) ──
        if (POLY_TRADING_ENABLED
                and poly_trader is not None
                and data.get('direction') not in ('NEUTRAL', 'OFFLINE')):
            asyncio.create_task(
                poly_trader.place_bet(
                    data['direction'], data['confidence'],
                    mahoraga_signal=_get_mahoraga_direction(),
                )
            )
        return JSONResponse(content=data)
    except Exception as e:
        log.error(f"[prediction] get_prediction_snapshot failed: {e}")
        return JSONResponse(content={
            "direction":                "NEUTRAL",
            "confidence":               0,
            "ob_imbalance":             0,
            "funding_rate":             0,
            "signal_raw":               "HOLD",
            "candle_close_countdown_s": int(900 - (time.time() % 900)),
            "timestamp":                datetime.now(timezone.utc).isoformat(),
            "status":                   "error",
            "error":                    str(e),
        })


@app.post("/api/autotrader/settings")
async def at_settings(_auth=Depends(require_auth)):
    # All risk parameters are hardcoded Ghost-algorithm constants.
    # They cannot be changed at runtime — returns the current constants.
    return JSONResponse(status_code=423, content={
        "error": "Risk parameters are hardcoded (Ghost-in-the-Market algorithm). They cannot be changed at runtime.",
        "risk_constants": autotrader.status()['risk_constants'],
    })
# ── REST: Virtual Trader ───────────────────────────────────────────
class VirtualBalanceUpdate(BaseModel):
    amount: float

@app.post("/api/virtual/toggle")
async def vt_toggle(_auth=Depends(require_auth)):
    if virtual_trader.running:
        virtual_trader.stop()
    else:
        virtual_trader.start()
    return {"status": "ok", "running": virtual_trader.running}

@app.post("/api/virtual/update_balance")
async def vt_update_balance(req: VirtualBalanceUpdate, _auth=Depends(require_auth)):
    virtual_trader.update_balance(req.amount)
    return {"status": "ok", "balance": virtual_trader.state['balance']}

@app.get("/api/virtual/status")
async def vt_status(_auth=Depends(require_auth)):
    return virtual_trader.status()

@app.get("/api/virtual/history")
async def vt_history(_auth=Depends(require_auth)):
    jp = os.path.join(_APP_DIR, 'trade_journal.json')
    if not os.path.exists(jp): return []
    try:
        with open(jp, 'r') as f:
            journal = json.load(f)
            return [t for t in journal if t.get('virtual', False)]
    except:
        return []


# ── REST: Polymarket ──────────────────────────────────────────────
_poly_status_call_count: int = 0

_POLY_OFFLINE = {
    "error":                  "Polymarket trader not initialized",
    "balance_usdc":           0,
    "active_bets":            [],
    "completed_bets":         [],
    "total_bets":             0,
    "total_won":              0,
    "total_lost":             0,
    "total_pnl":              0,
    "win_rate":               0.0,
    "adaptive_threshold_up":  80.0,
    "adaptive_threshold_down":80.0,
    "bet_candles_count":      0,
    "current_candle_locked":  False,
}

_poly_status_cache: dict | None = None   # last successful status response

@app.get("/api/polymarket/status")
async def polymarket_status(request: Request, _auth=Depends(require_auth)):
    global _poly_status_call_count, _poly_status_cache
    ip = request.client.host
    if _check_rate(ip, 'polymarket', POLYMARKET_RATE_LIMIT):
        # On rate limit, serve last known good status rather than failing
        if _poly_status_cache:
            return JSONResponse(content={**_poly_status_cache, "stale": True})
        return JSONResponse(status_code=429, content={"error": "Too many requests."})
    if poly_trader is None:
        return JSONResponse(content=_POLY_OFFLINE)
    try:
        _poly_status_call_count += 1
        # Refresh balance on first call and every 3rd call thereafter
        if _poly_status_call_count == 1 or _poly_status_call_count % 3 == 0:
            await poly_trader._refresh_balance()
        data = poly_trader.get_status()
        _poly_status_cache = data
        return JSONResponse(content=data)
    except Exception as e:
        log.error(f"[Polymarket] get_status failed: {e}")
        if _poly_status_cache:
            return JSONResponse(content={**_poly_status_cache, "stale": True})
        return JSONResponse(content=_POLY_OFFLINE)

@app.post("/api/polymarket/refresh-balance")
async def polymarket_refresh_balance(_auth=Depends(require_auth)):
    if poly_trader is None:
        return JSONResponse(content={"balance_usdc": 0})
    try:
        balance = await poly_trader._refresh_balance()
        return JSONResponse(content={"balance_usdc": balance})
    except Exception as e:
        log.error(f"[Polymarket] refresh-balance failed: {e}")
        return JSONResponse(content={"balance_usdc": poly_trader._wallet_balance})


@app.post("/api/polymarket/clear-history")
async def polymarket_clear_history(_auth=Depends(require_auth)):
    """Clear completed bet history. Active bets are preserved."""
    if poly_trader is None:
        return JSONResponse(content={"ok": False, "error": "Polymarket not available"})
    try:
        cleared = len(poly_trader.completed_bets)
        poly_trader.completed_bets.clear()
        poly_trader._save_bets()
        log.info(f"[Polymarket] Cleared {cleared} completed bets from history")
        return JSONResponse(content={"ok": True, "cleared": cleared})
    except Exception as e:
        log.error(f"[Polymarket] clear-history failed: {e}")
        return JSONResponse(content={"ok": False, "error": str(e)})


# ── Paper Trading Endpoints ────────────────────────────────────────

@app.get("/api/polymarket/paper/status")
async def paper_status(_auth=Depends(require_auth)):
    if poly_trader is None:
        return JSONResponse({"enabled": False, "balance": 0, "error": "Trader not initialized"})
    return JSONResponse(poly_trader.get_paper_status())

@app.post("/api/polymarket/paper/config")
async def paper_config_set(req: PaperConfigRequest, _auth=Depends(require_auth)):
    if poly_trader is None:
        return JSONResponse({"ok": False, "error": "Trader not initialized"})
    poly_trader._paper_mode = req.enabled
    if req.bet_size is not None and 0.01 <= req.bet_size <= 10000:
        poly_trader._paper_bet_size = round(req.bet_size, 2)
    if req.balance is not None and req.balance > 0:
        # Reset the paper account with fresh balance + clear history
        poly_trader._paper_balance  = req.balance
        poly_trader._paper_starting = req.balance
        poly_trader._paper_active.clear()
        poly_trader._paper_completed.clear()
        poly_trader._paper_candles.clear()
        poly_trader._save_paper_bets()
    poly_trader._save_paper_config()
    log.info(
        f"[PaperBet] Config updated — {'ON' if req.enabled else 'OFF'} | "
        f"balance=${poly_trader._paper_balance:.2f} | bet=${poly_trader._paper_bet_size:.2f}"
    )
    return JSONResponse({"ok": True, **poly_trader.get_paper_status()})

@app.get("/api/polymarket/paper/history")
async def paper_history_api(_auth=Depends(require_auth)):
    if poly_trader is None:
        return JSONResponse({"bets": [], "stats": {}})
    stats    = poly_trader.get_paper_status()
    all_bets = list(reversed(poly_trader._paper_completed)) + poly_trader._paper_active
    return JSONResponse({"bets": all_bets, "stats": stats})

@app.get("/polymarket/paper/history")
async def paper_history_page(request: Request, _auth=Depends(require_auth)):
    html = _build_paper_history_html()
    return HTMLResponse(content=html, headers={"Cache-Control": "no-store, no-cache"})

def _build_paper_history_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAHORAGA · Paper History</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#090909;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh;padding:24px 20px;}
.header{display:flex;align-items:center;gap:14px;margin-bottom:28px;padding-bottom:18px;border-bottom:1px solid #161616;}
.logo{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;letter-spacing:3px;color:#fff;}
.sep{color:#333;font-size:13px;}
.title{font-size:13px;color:#555;font-family:'JetBrains Mono',monospace;letter-spacing:1px;}
.badge{background:rgba(99,102,241,0.12);color:#6366f1;border:1px solid rgba(99,102,241,0.25);border-radius:4px;padding:2px 8px;font-size:10px;font-family:'JetBrains Mono',monospace;letter-spacing:1px;}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px;}
.stat-card{background:#111;border:1px solid #1a1a1a;border-radius:10px;padding:14px;}
.stat-label{font-size:10px;color:#444;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;}
.stat-val{font-size:1.3rem;font-weight:600;color:#fff;}
.stat-sub{font-size:10px;color:#333;margin-top:4px;font-family:'JetBrains Mono',monospace;}
.filters{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;}
.filter-btn{padding:5px 14px;border-radius:5px;border:1px solid #252525;background:transparent;color:#555;font-size:11px;font-family:'JetBrains Mono',monospace;cursor:pointer;transition:all .15s;letter-spacing:.5px;}
.filter-btn.active,.filter-btn:hover{border-color:#6366f1;color:#6366f1;background:rgba(99,102,241,0.08);}
.table-wrap{overflow:auto;border:1px solid #1a1a1a;border-radius:8px;background:#0a0a0a;}
table{width:100%;border-collapse:collapse;}
th{padding:9px 12px;text-align:left;font-size:10px;color:#333;font-family:'JetBrains Mono',monospace;letter-spacing:1px;border-bottom:1px solid #1a1a1a;white-space:nowrap;}
td{padding:9px 12px;font-size:11px;font-family:'JetBrains Mono',monospace;border-bottom:1px solid #0f0f0f;white-space:nowrap;}
tr:last-child td{border-bottom:none;}
tr:hover td{background:rgba(255,255,255,0.015);}
.dir-up{color:#00c896;font-weight:600;}
.dir-down{color:#ff4757;font-weight:600;}
.res-won{color:#00c896;font-weight:600;}
.res-lost{color:#ff4757;font-weight:600;}
.res-open{color:#f0a500;font-weight:600;}
.res-unk{color:#555;}
.pnl-pos{color:#00c896;}
.pnl-neg{color:#ff4757;}
.empty{padding:32px;text-align:center;color:#333;font-size:12px;}
.refresh-bar{display:flex;align-items:center;gap:12px;margin-bottom:16px;}
.refresh-dot{width:6px;height:6px;background:#6366f1;border-radius:50%;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
.ts{font-size:10px;color:#333;font-family:'JetBrains Mono',monospace;}
.question-col{max-width:220px;overflow:hidden;text-overflow:ellipsis;color:#555;}
</style>
</head>
<body>
<div class="header">
  <span class="logo">MAHORAGA</span>
  <span class="sep">/</span>
  <span class="title">PAPER TRADING HISTORY</span>
  <span class="badge">SIMULATION</span>
</div>

<div class="stat-grid" id="statsGrid">
  <div class="stat-card"><div class="stat-label">Paper Balance</div><div class="stat-val" id="s-balance">$--</div><div class="stat-sub" id="s-start">start: $--</div></div>
  <div class="stat-card"><div class="stat-label">Total P&L</div><div class="stat-val" id="s-pnl">—</div><div class="stat-sub" id="s-pnlpct">—%</div></div>
  <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-val" id="s-wr">—</div><div class="stat-sub" id="s-wl">0W · 0L</div></div>
  <div class="stat-card"><div class="stat-label">Total Bets</div><div class="stat-val" id="s-total">0</div><div class="stat-sub" id="s-active">active: 0</div></div>
  <div class="stat-card"><div class="stat-label">Bet Size</div><div class="stat-val" id="s-betsize">$--</div><div class="stat-sub">per simulation</div></div>
</div>

<div class="refresh-bar">
  <div class="refresh-dot"></div>
  <span class="ts" id="lastUpdate">Fetching…</span>
</div>

<div class="filters">
  <button class="filter-btn active" onclick="setFilter('all',this)">All</button>
  <button class="filter-btn" onclick="setFilter('open',this)">Active</button>
  <button class="filter-btn" onclick="setFilter('won',this)">Won</button>
  <button class="filter-btn" onclick="setFilter('lost',this)">Lost</button>
</div>

<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>DATE / TIME</th>
        <th>DIR</th>
        <th>CONF</th>
        <th>QUESTION</th>
        <th>ODDS</th>
        <th>SHARES</th>
        <th>COST</th>
        <th>EV</th>
        <th>RESULT</th>
        <th>P&L</th>
      </tr>
    </thead>
    <tbody id="tableBody">
      <tr><td colspan="11" class="empty">Loading…</td></tr>
    </tbody>
  </table>
</div>

<script>
let _allBets = [];
let _filter  = 'all';

function setFilter(f, btn) {
  _filter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderTable();
}

function renderTable() {
  const bets = _filter === 'all' ? _allBets
    : _filter === 'open' ? _allBets.filter(b => b.status === 'open')
    : _allBets.filter(b => b.result === _filter);
  const tbody = document.getElementById('tableBody');
  if (!bets.length) {
    tbody.innerHTML = '<tr><td colspan="11" class="empty">No trades match this filter</td></tr>';
    return;
  }
  tbody.innerHTML = bets.map((b, i) => {
    const dt  = b.placed_at ? new Date(b.placed_at).toLocaleString() : '--';
    const dir = b.direction === 'UP'
      ? '<span class="dir-up">▲ UP</span>'
      : '<span class="dir-down">▼ DOWN</span>';
    const resClass = b.status === 'open' ? 'res-open'
      : b.result === 'won' ? 'res-won'
      : b.result === 'lost' ? 'res-lost' : 'res-unk';
    const resText = b.status === 'open' ? 'OPEN'
      : b.result === 'won' ? 'WON'
      : b.result === 'lost' ? 'LOST' : 'UNKNOWN';
    const pnl = b.pnl !== null && b.pnl !== undefined ? b.pnl : null;
    const pnlStr = pnl !== null
      ? `<span class="${pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">${pnl >= 0 ? '+$' : '-$'}${Math.abs(pnl).toFixed(4)}</span>`
      : '<span style="color:#444">—</span>';
    const ev = b.ev !== undefined ? (b.ev >= 0 ? '+' : '') + (b.ev * 100).toFixed(1) + '%' : '—';
    const q  = (b.question || '').substring(0, 40) + ((b.question || '').length > 40 ? '…' : '');
    return `<tr>
      <td style="color:#333">${_allBets.length - i}</td>
      <td style="color:#444">${dt}</td>
      <td>${dir}</td>
      <td style="color:#666">${b.confidence ? b.confidence.toFixed(1) + '%' : '--'}</td>
      <td class="question-col" title="${b.question || ''}">${q}</td>
      <td style="color:#555">${b.price_paid ? b.price_paid.toFixed(3) : '--'}</td>
      <td style="color:#444">${b.shares || '--'}</td>
      <td style="color:#444">${b.bet_cost ? '$' + b.bet_cost.toFixed(4) : '--'}</td>
      <td style="color:${b.ev >= 0.10 ? '#00c896' : '#666'}">${ev}</td>
      <td><span class="${resClass}">${resText}</span></td>
      <td>${pnlStr}</td>
    </tr>`;
  }).join('');
}

async function fetchHistory() {
  try {
    const r = await fetch('/api/polymarket/paper/history', {credentials: 'include'});
    if (r.status === 401) { document.body.innerHTML = '<div style="padding:40px;text-align:center;color:#555;font-family:monospace;">Session expired — <a href="/" style="color:#6366f1;">Return to dashboard</a></div>'; return; }
    const d = await r.json();
    _allBets = d.bets || [];
    const s  = d.stats || {};

    // Stats
    const bal = s.balance || 0;
    const start = s.starting_balance || 100;
    const pnl = s.pnl || 0;
    const pnlPct = start > 0 ? (pnl / start * 100) : 0;
    document.getElementById('s-balance').textContent = '$' + bal.toFixed(2);
    document.getElementById('s-start').textContent   = 'start: $' + start.toFixed(2);
    const pnlEl = document.getElementById('s-pnl');
    pnlEl.textContent  = (pnl >= 0 ? '+$' : '-$') + Math.abs(pnl).toFixed(4);
    pnlEl.style.color  = pnl >= 0 ? '#00c896' : '#ff4757';
    const pctEl = document.getElementById('s-pnlpct');
    pctEl.textContent = (pnlPct >= 0 ? '+' : '') + pnlPct.toFixed(2) + '%';
    pctEl.style.color = pnlPct >= 0 ? '#00c896' : '#ff4757';
    const wrEl = document.getElementById('s-wr');
    wrEl.textContent = (s.total_won + s.total_lost > 0) ? s.win_rate + '%' : '—';
    wrEl.style.color = s.win_rate >= 55 ? '#00c896' : s.win_rate >= 45 ? '#f0a500' : '#ff4757';
    document.getElementById('s-wl').textContent    = `${s.total_won||0}W · ${s.total_lost||0}L`;
    document.getElementById('s-total').textContent = s.total_bets || 0;
    document.getElementById('s-active').textContent = 'active: ' + (s.active_bets ? s.active_bets.length : 0);
    document.getElementById('s-betsize').textContent = '$' + (s.bet_size || 1).toFixed(2);
    document.getElementById('lastUpdate').textContent = 'Updated: ' + new Date().toLocaleTimeString();
    renderTable();
  } catch(e) { console.error(e); }
}

fetchHistory();
setInterval(fetchHistory, 10000);
</script>
</body>
</html>"""

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
    if move_threshold < 0.001 or move_threshold > 0.20:
        raise HTTPException(400, "move_threshold must be between 0.001 and 0.20")
    import time as _time
    steps = []
    t0 = _time.time()
    try:
        steps.append("Fetching Bybit OHLCV data (1000 candles)…")
        df = fetch_bybit_data(symbol=symbol, interval=interval, limit=1000,
                              api_key=API_KEY, api_secret=API_SECRET)
        steps.append(f"Raw candles: {len(df)} rows")

        steps.append("Computing 18 features + forward-return labels…")
        X, y = preprocess_data(df, threshold=move_threshold)
        n_samples = int(len(y))
        n_buy  = int((y == 2).sum()) if hasattr(y, 'sum') else sum(1 for v in y if v == 2)
        n_sell = int((y == 0).sum()) if hasattr(y, 'sum') else sum(1 for v in y if v == 0)
        n_hold = n_samples - n_buy - n_sell
        steps.append(f"Samples: {n_samples}  (BUY={n_buy}  SELL={n_sell}  HOLD={n_hold})")

        steps.append("Applying Reinforcement Journal & retraining MLP (20 epochs)…")
        journal_data = autotrader._load_trade_journal()
        bot.retrain_with_journal(X, y, journal_data)

        steps.append("Saving model to MAHORAGA_model.pkl…")
        bot.save_model()
        # reset auto-retrain counter
        autotrader.trades_since_retrain = 0

        elapsed = round(_time.time() - t0, 1)
        steps.append(f"Done in {elapsed}s — model active")
        return {
            "status": "ok",
            "message": "MAHORAGA model trained and saved",
            "move_threshold": move_threshold,
            "samples": n_samples,
            "elapsed_secs": elapsed,
            "steps": steps,
        }
    except Exception as e:
        steps.append(f"ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e), "steps": steps})

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
        resp = await asyncio.to_thread(
            ac.messages.create,
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

                followup = await asyncio.to_thread(
                    ac.messages.create,
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


# ── REST: AI / continual-learning status ──────────────────────────
@app.get("/api/ai-status")
async def ai_status(_auth=Depends(require_auth)):
    """
    Returns live continual-learning metrics for the dashboard status bar:
      regime             — market regime from detect_regime() on the last df
      last_retrain_iso   — ISO-8601 timestamp of last successful retrain, or "never"
      rolling_win_rate   — win rate over the last 20 closed trades (0–100)
      trades_since_retrain — count of trades since last retrain
    """
    last_df = autotrader._last_df
    regime = 'unknown'
    if last_df is not None and not last_df.empty:
        try:
            regime = await asyncio.to_thread(bot.detect_regime, last_df)
        except Exception:
            pass

    lrt = autotrader._last_retrain_time
    if lrt == 0:
        last_retrain_iso = 'never'
    else:
        last_retrain_iso = datetime.fromtimestamp(lrt, tz=timezone.utc).isoformat()

    outcomes = autotrader._rolling_outcomes
    rolling_win_rate = (
        round(outcomes.count('win') / len(outcomes) * 100, 1)
        if outcomes else 0.0
    )

    # ── Auto-retrain version info ──────────────────────────────────
    best_ver, best_acc   = get_best_model_version()
    history              = _load_version_history()
    live_entries         = [e for e in history if e.get('is_live')]
    cur_ver              = live_entries[-1].get('version', 0) if live_entries else 0
    cur_acc              = live_entries[-1].get('val_accuracy', bot.current_accuracy) \
                           if live_entries else bot.current_accuracy
    next_retrain_in      = max(0, AUTO_RETRAIN_EVERY - autotrader.trades_since_retrain)
    meta                 = autotrader._last_retrain_meta

    return JSONResponse(content={
        'regime':               regime,
        'last_retrain_iso':     last_retrain_iso,
        'rolling_win_rate':     rolling_win_rate,
        'trades_since_retrain': autotrader.trades_since_retrain,
        'auto_retrain': {
            'enabled':             True,
            'trades_since_retrain': autotrader.trades_since_retrain,
            'next_retrain_in':     next_retrain_in,
            'current_version':     cur_ver,
            'current_accuracy':    round(float(cur_acc), 4),
            'best_version':        best_ver,
            'best_accuracy':       round(float(best_acc), 4),
            'is_retraining':       autotrader._retraining,
            'last_retrain':        meta,
        },
    })


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

    async def startup_and_run():
        global _startup_task
        # Start the automatic startup scheduler
        _startup_task = asyncio.create_task(_auto_startup_scheduler())
        log.info(f"Starting MAHORAGA on {BIND_HOST}:{BIND_PORT}")
        if not DASHBOARD_PASSWORD:
            log.warning("⚠  Running without a password — set DASHBOARD_PASSWORD in .env")
        
        # Configure uvicorn to run with asyncio
        config = uvicorn.Config(app, host=BIND_HOST, port=BIND_PORT, reload=False)
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(startup_and_run())
