import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Paths
ROOT          = Path(__file__).parent.parent
MODEL_PATH    = ROOT / "STORM_v1.joblib"
CSV_PATH      = ROOT / "STORM.csv"
LOG_FILE      = ROOT / "bot" / "bet_log.jsonl"

# Polymarket
POLY_API_KEY        = os.getenv("POLY_API_KEY")
POLY_API_SECRET     = os.getenv("POLY_API_SECRET")
POLY_API_PASSPHRASE = os.getenv("POLY_API_PASSPHRASE")
POLY_PRIVATE_KEY    = os.getenv("POLY_PRIVATE_KEY")
POLY_FUNDER         = os.getenv("POLY_FUNDER")
POLY_CHAIN_ID       = int(os.getenv("POLY_CHAIN_ID", "137"))

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

# Betting controls
DAILY_BUDGET_USDC = float(os.getenv("DAILY_BUDGET_USDC", "1.0"))
MIN_EDGE          = float(os.getenv("MIN_EDGE",          "0.05"))

# Scheduler
SCHEDULE_TIME = os.getenv("SCHEDULE_TIME", "20:00")
