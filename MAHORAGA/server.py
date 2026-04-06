import os, json, asyncio, logging, re, time
from collections import defaultdict
from datetime import datetime, timezone
import anthropic
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from core_trading_system import MAHORAGA, fetch_bybit_data, get_bybit_client, preprocess_data

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

API_KEY       = os.getenv('BYBIT_API_KEY', '')
API_SECRET    = os.getenv('BYBIT_API_SECRET', '')
CHAT_PASSWORD = os.getenv('CHAT_PASSWORD', '')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('MAHORAGA')

# ── CHAT SECURITY ─────────────────────────────────────────────────
CHAT_RATE_LIMIT  = 15          # max messages per minute per IP
MAX_MSG_LEN      = 600         # max characters per user message
_chat_rate: dict = defaultdict(list)

_STRIP_HTML = re.compile(r'<[^>]+>')

# This system prompt is server-side and cannot be overridden by the frontend.
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

app = FastAPI()
bot = MAHORAGA()

# ── AUTO TRADER ────────────────────────────────────────────────────
class AutoTrader:
    SYMBOL   = 'BTCUSDT'
    INTERVAL = '60'
    MIN_QTY  = 0.001

    def __init__(self):
        self.running         = False
        self.task            = None
        self.check_secs      = 300
        self.last_signal     = None
        self.next_check      = None
        self.trade_log       = []
        # Risk settings
        self.conf_threshold  = 0.65
        self.sl_pct          = 1.5
        self.tp_pct          = 3.0
        self.trailing_pct    = 1.0
        self.max_daily_loss  = 10.0
        self._daily_loss_ref = None

    # ── helpers ──
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
            qty   = round((bal * 0.10) / price, 3)
            return max(self.MIN_QTY, qty)
        except Exception:
            return self.MIN_QTY

    def _get_wallet(self, client):
        w = client.get_wallet_balance(accountType='UNIFIED', coin='USDT')
        return float(w['result']['list'][0]['coin'][0]['walletBalance'])

    def _sl_tp_prices(self, client, side):
        t     = client.get_tickers(category='linear', symbol=self.SYMBOL)
        price = float(t['result']['list'][0]['markPrice'])
        sl_dist = price * (self.sl_pct / 100)
        tp_dist = price * (self.tp_pct / 100)
        if side == 'Buy':
            return round(price - sl_dist, 2), round(price + tp_dist, 2)
        else:
            return round(price + sl_dist, 2), round(price - tp_dist, 2)

    def _trailing_distance(self, client):
        t     = client.get_tickers(category='linear', symbol=self.SYMBOL)
        price = float(t['result']['list'][0]['markPrice'])
        return str(round(price * (self.trailing_pct / 100), 2))

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

    def _check_daily_loss(self, client):
        try:
            bal = self._get_wallet(client)
            if self._daily_loss_ref is None:
                self._daily_loss_ref = bal
                return False
            loss_pct = (self._daily_loss_ref - bal) / self._daily_loss_ref * 100
            return loss_pct >= self.max_daily_loss
        except Exception:
            return False

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

    # ── main loop ──
    async def run(self):
        while self.running:
            self.next_check = datetime.now().timestamp() + self.check_secs
            try:
                client = self._client()

                # ── 0. Protect any unguarded existing position ──
                position = self._get_position(client)
                if position and not position.get('stopLoss') and not position.get('takeProfit'):
                    try:
                        t     = client.get_tickers(category='linear', symbol=self.SYMBOL)
                        mark  = float(t['result']['list'][0]['markPrice'])
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
                                  f'SL=${sl} TP=${tp} Trail={self.trailing_pct}%')
                    except Exception as eg:
                        self._log('GUARD', 0, 'FAILED TO PROTECT POSITION', str(eg)[:80])

                # ── 1. Max daily loss check ──
                if self._check_daily_loss(client):
                    self._log('RISK', 0, 'STOPPED — MAX DAILY LOSS',
                              f'{self.max_daily_loss}% drawdown hit. Bot paused.')
                    self.stop()
                    break

                # ── 2. Get AI signal ──
                df       = fetch_bybit_data(symbol=self.SYMBOL, interval=self.INTERVAL,
                                            limit=300, api_key=API_KEY, api_secret=API_SECRET)
                features = bot.preprocess_single_bar(df.copy())
                signal, confidence = bot.predict(features, self.conf_threshold)
                self.last_signal = {
                    'signal':     signal,
                    'confidence': round(float(confidence), 4),
                    'time':       datetime.now().strftime('%H:%M:%S'),
                }

                # ── 3. Get current position ──
                position = self._get_position(client)
                pos_side = position['side'] if position else None

                # ── 4. Trade decision ──
                if confidence < self.conf_threshold or signal == 'HOLD':
                    self._log(signal, confidence, 'SKIP',
                              f'conf={round(confidence*100,1)}% — HOLD or below threshold')

                elif signal == 'BUY':
                    if pos_side == 'Buy':
                        self._log(signal, confidence, 'SKIP', 'already long')
                    else:
                        if pos_side == 'Sell':
                            cr = self._place(client, 'Buy', float(position['size']), reduce_only=True)
                            if cr.get('retCode') != 0:
                                self._log(signal, confidence, 'FAILED', f"close short: {cr.get('retMsg')}")
                                await asyncio.sleep(self.check_secs); continue
                            self._log(signal, confidence, 'CLOSED SHORT', cr['result'].get('orderId','')[:8])
                            await asyncio.sleep(1)
                        sl, tp       = self._sl_tp_prices(client, 'Buy')
                        trail        = self._trailing_distance(client)
                        qty          = self._get_qty(client)
                        r            = self._place(client, 'Buy', qty, sl=sl, tp=tp, trailing=trail)
                        if r.get('retCode') == 0:
                            self._log(signal, confidence,
                                      f'OPENED LONG {qty} BTC',
                                      f'SL=${sl} TP=${tp} Trail={self.trailing_pct}% | {r["result"].get("orderId","")[:8]}')
                        else:
                            self._log(signal, confidence, 'FAILED LONG', r.get('retMsg',''))

                elif signal == 'SELL':
                    if pos_side == 'Sell':
                        self._log(signal, confidence, 'SKIP', 'already short')
                    else:
                        if pos_side == 'Buy':
                            cr = self._place(client, 'Sell', float(position['size']), reduce_only=True)
                            if cr.get('retCode') != 0:
                                self._log(signal, confidence, 'FAILED', f"close long: {cr.get('retMsg')}")
                                await asyncio.sleep(self.check_secs); continue
                            self._log(signal, confidence, 'CLOSED LONG', cr['result'].get('orderId','')[:8])
                            await asyncio.sleep(1)
                        sl, tp       = self._sl_tp_prices(client, 'Sell')
                        trail        = self._trailing_distance(client)
                        qty          = self._get_qty(client)
                        r            = self._place(client, 'Sell', qty, sl=sl, tp=tp, trailing=trail)
                        if r.get('retCode') == 0:
                            self._log(signal, confidence,
                                      f'OPENED SHORT {qty} BTC',
                                      f'SL=${sl} TP=${tp} Trail={self.trailing_pct}% | {r["result"].get("orderId","")[:8]}')
                        else:
                            self._log(signal, confidence, 'FAILED SHORT', r.get('retMsg',''))

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log('ERROR', 0, 'EXCEPTION', str(e)[:120])

            await asyncio.sleep(self.check_secs)

    def start(self):
        if not self.running:
            self.running         = True
            self._daily_loss_ref = None
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
        return {
            'running':        self.running,
            'symbol':         self.SYMBOL,
            'interval':       self.INTERVAL,
            'check_secs':     self.check_secs,
            'next_check':     secs_left,
            'last_signal':    self.last_signal,
            'log':            self.trade_log[:20],
            'settings': {
                'conf_threshold': self.conf_threshold,
                'sl_pct':         self.sl_pct,
                'tp_pct':         self.tp_pct,
                'trailing_pct':   self.trailing_pct,
                'max_daily_loss': self.max_daily_loss,
            }
        }

autotrader = AutoTrader()

# ── Dashboard HTML ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "MAHORAGA_dashboard.html")
    with open(html_path, "r") as f:
        return f.read()

# ── REST: market snapshot ─────────────────────────────────────────
@app.get("/api/market")
def market(symbol: str = "BTCUSDT", interval: str = "60", limit: int = 200):
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

# ── REST: AI signal ───────────────────────────────────────────────
@app.get("/api/signal")
def signal(symbol: str = "BTCUSDT", interval: str = "60", confidence: float = 0.6):
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

# ── REST: balance ─────────────────────────────────────────────────
@app.get("/api/balance")
def balance():
    try:
        bal = bot.get_balance(API_KEY, API_SECRET)
        return {"usdt": round(bal, 2)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── REST: full dashboard data ──────────────────────────────────────
@app.get("/api/dashboard")
def dashboard_data(symbol: str = "BTCUSDT"):
    try:
        client = get_bybit_client(API_KEY, API_SECRET)

        w    = client.get_wallet_balance(accountType='UNIFIED', coin='USDT')
        coin = w['result']['list'][0]['coin'][0]

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

        return {
            'walletBalance':  round(float(coin['walletBalance']), 4),
            'equity':         round(float(coin['equity']), 4),
            'unrealisedPnl':  round(float(coin['unrealisedPnl']), 4),
            'cumRealisedPnl': round(float(coin['cumRealisedPnl']), 4),
            'todayPnl':       round(today_pnl, 4),
            'todayTrades':    len(today_ex),
            'totalTrades':    len(execs),
            'positions':      positions,
            'autotrader':     autotrader.status(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── REST: place order ─────────────────────────────────────────────
@app.post("/api/order")
def order(symbol: str = "BTCUSDT", side: str = "Buy", qty: float = 0.001):
    try:
        result = bot.place_order(symbol, side, qty, API_KEY, API_SECRET)
        if result.get("retCode") != 0:
            msg = result.get("retMsg", "Unknown error")
            return JSONResponse(status_code=400, content={"error": msg, "retCode": result.get("retCode")})
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── REST: autotrader controls ──────────────────────────────────────
@app.post("/api/autotrader/start")
async def at_start():
    if not bot.model:
        return JSONResponse(status_code=400, content={"error": "No model loaded. Train first."})
    autotrader.start()
    return {"status": "started"}

@app.post("/api/autotrader/stop")
async def at_stop():
    autotrader.stop()
    return {"status": "stopped"}

@app.get("/api/autotrader/status")
async def at_status():
    return autotrader.status()

class ATSettings(BaseModel):
    sl_pct:          float = None
    tp_pct:          float = None
    trailing_pct:    float = None
    max_daily_loss:  float = None
    conf_threshold:  float = None

@app.post("/api/autotrader/settings")
async def at_settings(s: ATSettings):
    if s.sl_pct         is not None: autotrader.sl_pct         = s.sl_pct
    if s.tp_pct         is not None: autotrader.tp_pct         = s.tp_pct
    if s.trailing_pct   is not None: autotrader.trailing_pct   = s.trailing_pct
    if s.max_daily_loss is not None: autotrader.max_daily_loss = s.max_daily_loss
    if s.conf_threshold is not None: autotrader.conf_threshold = s.conf_threshold
    return autotrader.status()['settings']

# ── REST: train model ─────────────────────────────────────────────
@app.post("/api/train")
def train(symbol: str = "BTCUSDT", interval: str = "60"):
    try:
        df = fetch_bybit_data(symbol=symbol, interval=interval, limit=1000,
                              api_key=API_KEY, api_secret=API_SECRET)
        X, y = preprocess_data(df)
        bot.train_model(X, y)
        bot.save_model()
        return {"status": "ok", "message": "MAHORAGA model trained and saved"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── REST: chat ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    messages: List[dict]
    context:  str = ""

def _close_open_position() -> dict:
    """Close any open BTCUSDT position with a reduce-only market order."""
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
            log.info(f"[Chat] close_position executed: side={close_side} size={size} retCode={r.get('retCode')}")
            return r
    return {"retCode": -1, "retMsg": "No open position found"}

@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    # ── Password check ──
    if CHAT_PASSWORD:
        token = request.headers.get("X-Chat-Token", "")
        if token != CHAT_PASSWORD:
            log.warning(f"[Chat] Unauthorised attempt from {request.client.host}")
            return JSONResponse(status_code=401, content={"error": "Unauthorised"})

    # ── Rate limiting ──
    ip  = request.client.host
    now = time.time()
    _chat_rate[ip] = [t for t in _chat_rate[ip] if now - t < 60]
    if len(_chat_rate[ip]) >= CHAT_RATE_LIMIT:
        return JSONResponse(status_code=429, content={"error": "Too many messages. Wait a moment."})
    _chat_rate[ip].append(now)

    # ── Sanitise & validate input ──
    def clean(text: str) -> str:
        return _STRIP_HTML.sub('', text).strip()[:MAX_MSG_LEN]

    messages = [
        {"role": m["role"], "content": clean(m.get("content", ""))}
        for m in req.messages[-10:]
        if m.get("role") in ("user", "assistant") and m.get("content", "").strip()
    ]
    if not messages:
        return JSONResponse(status_code=400, content={"error": "Empty message."})

    # ── Build system: security prompt + live context ──
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

        # ── Handle tool call ──
        if resp.stop_reason == "tool_use":
            tool_block = next((b for b in resp.content if b.type == "tool_use"), None)
            if tool_block and tool_block.name == "close_position":
                close_result = await asyncio.to_thread(_close_open_position)
                success      = close_result.get("retCode") == 0
                tool_result  = "Position closed successfully." if success else f"Failed: {close_result.get('retMsg', 'unknown error')}"

                # Feed result back so Claude can confirm in natural language
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

# ── WebSocket: live feed ──────────────────────────────────────────
@app.websocket("/ws/price")
async def ws_price(ws: WebSocket):
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
            except Exception:
                pass
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8501, reload=False)
