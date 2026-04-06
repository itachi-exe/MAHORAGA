# MAHORAGA — AI Trading Bot

## Quick Start

### 1. Install dependencies (first time only)
```bash
cd /home/d3f4ult/Desktop/MAHORAGA
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn python-dotenv anthropic pybit pandas numpy scikit-learn ta joblib pydantic websockets httpx
```

### 2. Configure your keys
Edit the `.env` file:
```
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret
ANTHROPIC_API_KEY=your_anthropic_api_key
DASHBOARD_PASSWORD=your_dashboard_password
```

### 3. Start the bot
```bash
cd /home/d3f4ult/Desktop/MAHORAGA
source .venv/bin/activate
python server.py
```

### 4. Open the dashboard
Open your browser and go to:
```
http://127.0.0.1:8501
```
Enter your `DASHBOARD_PASSWORD` to unlock.

---

## Stopping the bot
Press `Ctrl + C` in the terminal where the server is running.

---

## Restarting after reboot
```bash
cd /home/d3f4ult/Desktop/MAHORAGA
source .venv/bin/activate
python server.py
```

---

## Train the AI model (if no model exists)
1. Open the dashboard → go to **Train** section
2. Click **Train Model**
3. Wait ~30 seconds — model saved automatically

---

## Auto Trader
- Go to dashboard → **Auto Trader** tab
- Click **Start** to enable automated trading
- Click **Stop** to pause
- Checks for signals every **5 minutes**

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Module not found` | Run `pip install -r` or reinstall deps (Step 1) |
| Dashboard won't load | Make sure server is running on port 8501 |
| Login fails | Check `DASHBOARD_PASSWORD` in `.env` |
| No signal / model error | Train the model first from the dashboard |
| Bybit API error | Check API key permissions — needs Futures trading enabled |

---

## Security Notes
- Dashboard is locked behind a password (login overlay)
- Server only listens on `127.0.0.1` by default (localhost only)
- To expose on your local network, set `BIND_HOST=0.0.0.0` in `.env`
- All API endpoints require an active session — no unauthenticated access

---

## File Structure
```
MAHORAGA/
├── server.py                  # Main server (FastAPI)
├── core_trading_system.py     # AI model + Bybit client
├── MAHORAGA_dashboard.html    # Frontend dashboard
├── MAHORAGA_model.pkl         # Trained AI model
├── MAHORAGA_scaler.pkl        # Feature scaler
├── .env                       # API keys & config
└── .venv/                     # Python virtual environment
```
